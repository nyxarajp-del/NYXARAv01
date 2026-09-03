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
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from dataclasses import field as dc_field
from typing import Any, Dict, List, Optional, Tuple

from nyxara.njp.fabric import Fabric, GrowthReport, SettleResult
from nyxara.njp.ledger import Ledger
from nyxara.njp.manifold import Prediction
from nyxara.njp.soulsync import Reading, SoulSync
from nyxara.njp.truth import Judgement, TruthGauntlet

log = logging.getLogger("nyxara.njp.brain")

#: When the fabric's recognition of a turn counts as "I have not met this", and when an answer
#: counts as confident. Both floors are needed together: unfamiliar ground is most turns early on
#: and a confident answer is the ordinary case, and it is the combination that is worth asking
#: about — sure of herself on ground she does not recognise.
_ANOMALY_FAMILIARITY = 0.2
_ANOMALY_CONFIDENCE = 0.7

#: What a claim may still be worth once the substrate names a different filler for its slot. A cap
#: rather than a retraction: the fabric associates and is measured at MRR 0.238, so it may make her
#: less sure and may never take a grounded answer away.
_SUBSTRATE_CEILING = 0.5

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
    #: What :mod:`nyxara.njp.discourse` made of this turn — the act behind it, what its pronouns
    #: name, what it does to the claims already on the ledger, and whether it is a sentence to be
    #: taken literally at all. A ``discourse.Uptake``; ``None`` when the organ is off.
    #:
    #: It sits on the **percept** rather than on the thought because two things downstream of
    #: perception need it before a thought exists: the sentence handed to the grounder has its
    #: settled pronouns already filled in, and a sentence read as non-literal never reaches the
    #: store at all.
    uptake: Any = None

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
    #: The ranked field of accounts for this turn — see :mod:`nyxara.njp.compete`.
    competition: Any = None
    #: The genome's record of this turn's reasoning, kept so the outcome can be graded against
    #: the *shape* that produced it once reality says whether it was right.
    trace: Any = None
    # What the Master was *doing* with this turn, and what each recalled memory scored against
    # it. Carried so a reply that reached for something irrelevant is visible from outside rather
    # than having to be inferred from the reply itself.
    act: Any = None
    #: This turn compiled into one executable operation — a ``njp.compile.CognitiveOp``. The
    #: three readings of a turn (intent, speech act, pattern match) meet here and nowhere else;
    #: before it they never met at all, and the one that could reach the causal engine was the
    #: one that knew least.
    op: Any = None
    #: Set when the symbolic and neural seats named different fillers for one slot — a
    #: ``router.DisagreementKind``. ``None`` means they agreed, or only one of them spoke, and
    #: those are deliberately the same thing here: neither is a clash.
    substrate: Any = None
    #: The worked solution when this turn was a mathematics task — a ``mathematics.Solution``,
    #: carrying the topic, the method and the steps. Kept beside the answer rather than folded
    #: into it for the reason `derivation` is: a computed answer and a recalled one read the same
    #: in a string and are entirely different epistemic objects.
    mathematics: Any = None
    #: The reading :mod:`nyxara.njp.corpussolver` made of this turn — a ``corpussolver.Reading``,
    #: carrying which engine claimed it, why, and whether a second independent computation agreed.
    #: Kept beside the answer for the reason ``mathematics`` is, plus one of its own: ``verified``
    #: is the difference between an answer that was checked twice and one that was not, and that
    #: distinction is destroyed by folding it into a string.
    cognition: Any = None
    #: The retirement trial run this turn, where the gate is on — a ``field.Trial``. ``None``
    #: means the gate is off or the slow count was not due, and those are both "nothing was
    #: proposed" rather than "a proposal was refused".
    retirement: Any = None
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
    # NJP V.06 — the cortex consult. All three are ``None`` when the cortex is off or unreachable,
    # which is deliberately different from "it was consulted and said nothing".
    #: Which seats this turn permitted, in which order, and on what basis. A ``Routing``.
    routing: Any = None
    #: What her cortex proposed and what the gates did with it. A ``CortexReport``.
    cortex: Any = None
    #: What the two seats came to — including, when they came to nothing, the question that
    #: would settle it. An ``Arbitration``.
    arbitration: Any = None
    #: The discriminating experiment compiled out of a disagreement. An ``Experiment``.
    experiment: Any = None
    #: Where this turn's answer came from — see :class:`~nyxara.njp.grounding.Provenance`. Carried
    #: on the thought so a caller can tell a witnessed answer from a proposed one without having to
    #: reach back into the grounding result.
    provenance: str = "observed"

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
        # `confidence` and `epistemic_confidence` are different questions and a report that shows
        # only the first is misread every time. `confidence` is the *gauntlet's* number: it is
        # 0.0 whenever the verdict was `abstained`, which is the correct, fail-closed outcome for
        # a claim with no hard evidence source — "Jay" has none, and never will. Meanwhile the
        # answer itself was `believed` at 0.9. Reading the pair as a contradiction is what an
        # audit did, and it cost a round trip to establish that nothing was broken.
        #
        # So both travel, each labelled with what it is about: what she may *state as fact*, and
        # how sure she is of what she *said*.
        return {"stimulus": self.stimulus[:200], "answer": self.answer[:400],
                "cycle_id": self.cycle_id, "confidence": round(self.confidence, 4),
                "epistemic": self.epistemic,
                "epistemic_confidence": round(self.epistemic_confidence, 4),
                "novelty": (round(self.percept.novelty, 4) if self.percept else None),
                "verified": self.verified, "assertable": self.assertable,
                "ms": round(self.ms, 3),
                "percept": self.percept.to_dict() if self.percept else None,
                "reading": self.reading.to_dict() if self.reading else None,
                "intent": self.intent.to_dict() if hasattr(self.intent, "to_dict") else None,
                "judgement": self.judgement.to_dict() if self.judgement else None,
                "growth": self.growth.to_dict() if self.growth else None,
                "loop": self.loop.to_dict() if self.loop is not None else None,
                "provenance": self.provenance,
                "routing": self.routing.to_dict() if self.routing is not None else None,
                "cortex": self.cortex.to_dict() if self.cortex is not None else None,
                "arbitration": self.arbitration.to_dict() if self.arbitration is not None else None,
                "experiment": self.experiment.to_dict() if self.experiment is not None else None,
                "field": self.field.to_dict() if self.field is not None else None,
                "trial": self.trial.to_dict() if self.trial is not None else None,
                # The working, when this turn was worked out. Carried over the wire for the same
                # reason it is carried on the thought: "6" and "6, because hcf(48,18) is 6" are
                # the same answer and very different evidence, and a caller that cannot see the
                # second cannot tell a computation from a recollection.
                "mathematics": (self.mathematics.to_dict()
                                if self.mathematics is not None else None),
                "focus": self.focus.to_dict() if self.focus is not None else None}


def _confirm(triple: Any) -> str:
    """One stored triple, said back **with its polarity**.

    Separate from :meth:`NJPBrain._compose` so it can be tested directly, and because it is the
    whole of the V.26 store defect: a confirmation that drops the negator is not a shorter
    confirmation, it is a different claim.
    """
    try:
        relation = str(getattr(triple, "predicate", "")).replace("_", " ")
        core = f"{getattr(triple, 'subject', '')} {relation} {getattr(triple, 'object', '')}"
        if getattr(triple, "negated", False):
            return f"{getattr(triple, 'subject', '')} does not {relation} " \
                   f"{getattr(triple, 'object', '')}".strip()
        return core.strip()
    except Exception:  # noqa: BLE001
        return ""


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
        #: cell id → the concept label that minted it. The inverse of :func:`_cell_id`, which is a
        #: one-way digest — so this is the only thing that lets the substrate name anything it
        #: predicts. Written by :meth:`encode` and persisted with the fabric, because a reloaded
        #: brain that cannot name its own cells is exactly the failure `_cell_id`'s own docstring
        #: warns about, one layer up.
        self.lexicon: Dict[int, str] = {}
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
        # After the grounder, because it reads nothing from it, and before `perceive` can run.
        self.discourse = self._build_discourse(c)
        self.world = self._build_world(c)
        self.predictor = self._build_predictor(c)
        self.levels = self._build_levels(c)
        self.discoverer = self._build_discoverer(c)
        self.reasoner = self._build_reasoner(c)
        self.self_model = self._build_self_model(c)
        self.genome = self._build_genome(c)
        # Before `metareason`, which registers a strategy bound to it: a calculator built after
        # the strategy table would be registered as absent and never chosen.
        self.calculator = self._build_calculator(c)
        # After the calculator and before anything that reads a turn: the mathematician borrows
        # the calculator rather than holding a second one, and `perceive` asks it *first* — before
        # the grounder — so a maths instruction is never filed as a fact about the world.
        self.mathematician = self._build_mathematician(c)
        # After the mathematician and consulted *before* it: see `_mathematical`.
        self.solver = self._build_solver(c)
        # After the mathematician's solver and consulted *before* it: see `_cognitive`. Its
        # readings each require a whole structure — a chain with a modulus, a rule base with a
        # goal, a dependency graph — so it claims nothing the mathematician wants.
        self.cognition = self._build_cognition(c)
        self.meta = self._build_meta(c)
        self.goals = self._build_goals(c)
        self.curiosity = self._build_curiosity(c)
        self.assumptions = self._build_assumptions(c)
        self.blackbox = self._build_blackbox(c)
        self.evolution = self._build_evolution(c)
        self.doing = self._build_doing(c)
        self.index = self._build_index(c)
        self.society = self._build_society(c)
        #: The chain `_strategy_compose` walked on this turn, if it walked one. Set by the
        #: strategy and read by the turn that called it — `MetaReasoner.solve` copies its
        #: context, so there is no other route back.
        self._composition: Any = None
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
        # The one place the three readings of a turn meet. Before `universe`, because the
        # counterfactual context is compiled through it and the universe is what executes it.
        self.compiler = self._build_compiler(c)
        self.universe = self._build_universe(c)
        # After `universe`, whose model it rolls forward, and after `predictor`, which is where
        # its claims go to be scored. A planner built before either would hold two Nones and
        # report every search as "no causal model".
        self.rollout = self._build_rollout(c)
        self.designer = self._build_designer(c)
        self.beliefs = self._build_beliefs(c)
        # After `beliefs`, whose claims it hunts, and after `universe`; it also reads `grounder`
        # and `world`, both built well above. A killer with no ledger has nothing to go after.
        self.killer = self._build_killer(c)
        # After `curriculum` is *defined* but read lazily — the generator holds the same stage
        # list and reads `brain.stats()`, so it needs no organ handed to it at construction.
        self.proposer = self._build_proposer(c)
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
        # ---- NJP V.06: the cortex, and the two organs that keep it honest ---- #
        # Built here, after `metareason`, `self_model`, `adversary` and `curiosity`, because every
        # one of them is a collaborator rather than a copy: the router asks the classifier what
        # kind of problem this is, weighs the seats by the self-model's posteriors, and the cortex's
        # laws are attacked before they are believed. A cortex wired to none of them would be a
        # second opinion with nothing to check it — which is the arrangement that produced weights
        # with no capability behind them.
        self.cortex = self._build_cortex(c)
        self.router = self._build_router(c)
        self.epistemic = self._build_epistemic(c)
        self.competition = self._build_competition(c)
        self.embedding = self._build_embedding(c)
        # The predictive brain: what follows what in the WORLD, as opposed to which of her own
        # cells fire next. Without it she is a fact store that grows; with it she is something
        # that expects, is wrong, and learns from the difference.
        self.predictive = self._build_predictive(c)
        # Stage G: she plans over what she has actually learned, acts, and is graded by the
        # world. Built after `predictive` because it plans by searching that model — an agent
        # with no world model does not plan, it flails.
        self.agent = self._build_agent(c)
        # Held, not driven: the index reads its curve, a caller steps it deliberately.
        self.noesis = self._build_noesis(c)
        self.curriculum = self._build_curriculum(c)
        # The programming faculty. Late, and dependent on nothing above it: a Coder holds shapes
        # and an interpreter, and neither reads another organ. It is here rather than at the top
        # because the gate should be able to switch it off without disturbing anything that runs
        # before it.
        self.coder = self._build_coder(c)
        # The language faculty, and the one edge from it into what she believes. After the
        # grounder because it is handed to the grounder, and it is handed over rather than
        # consulted from here so that the *reading* of a sentence stays in one place: a brain
        # that parsed a turn twice, once for structure and once for the grammar, would have two
        # answers to "what did he just say" and no rule for which one wins.
        self.language = self._build_language(c)
        if self.grounder is not None and self.language is not None:
            try:
                self.grounder.grammar = self.language
            except Exception:  # noqa: BLE001 — she grounds exactly as before without it
                pass
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

    def _build_genome(self, c: Any) -> Any:
        """The record of how she reasons, so a shape she keeps re-deriving can be seen."""
        if not self._gate("genome", True):
            return None
        try:
            from nyxara.njp.genome import ReasoningGenome
            return ReasoningGenome()
        except Exception:  # noqa: BLE001
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

    def _build_proposer(self, c: Any) -> Any:
        """The rung she is not on, proposed from measured weakness rather than read off a list.

        The ladder in :mod:`nyxara.njp.curriculum` is fixed and walked in order, so one rung
        blocked on a sample count pins the whole thing while later rungs are already mastered.
        See :mod:`nyxara.njp.propose`.
        """
        if not self._gate("propose", True):
            return None
        try:
            from nyxara.njp.propose import StageGenerator
            return StageGenerator(self)
        except Exception:  # noqa: BLE001 — she works the fixed ladder, as before
            return None

    def _build_killer(self, c: Any) -> Any:
        """What would end what she is most committed to — and has it already happened?

        Every belief carried a falsifier and every falsifier was true of everything, because one
        template wrote all of them and nothing ever went looking for what they named. See
        :mod:`nyxara.njp.falsify`.
        """
        if not self._gate("falsify", True):
            return None
        try:
            from nyxara.njp.falsify import TheoryKiller
            return TheoryKiller(self)
        except Exception:  # noqa: BLE001 — her beliefs then only ever gain support, as before
            return None

    def _build_rollout(self, c: Any) -> Any:
        """Goal → imagined futures → the best one → and reality's verdict on it.

        :meth:`~nyxara.njp.universe.InternalUniverse.imagine` had exactly one caller, which
        imagined one variable at the value it already had. Nothing asked which intervention would
        get a number where it needed to be, which is the only question a causal model is for.
        See :mod:`nyxara.njp.rollout`.
        """
        if not self._gate("rollout", True):
            return None
        try:
            from nyxara.njp.rollout import RolloutPlanner
            return RolloutPlanner(self)
        except Exception:  # noqa: BLE001 — she models what follows what and plans nothing with it
            return None

    def _build_assumptions(self, c: Any) -> Any:
        """What every arrow she believes is resting on, and which of it nothing has examined.

        Phase 3's missing half. She could say what she knows and what she does not; neither
        answers the question that actually kills a model — *which assumptions have I never
        tested?* See :mod:`nyxara.njp.assume`.
        """
        if not self._gate("assumptions", True):
            return None
        try:
            from nyxara.njp.assume import AssumptionMiner
            return AssumptionMiner()
        except Exception:  # noqa: BLE001 — she reasons on unexamined arrows, as she always did
            return None

    def _build_blackbox(self, c: Any) -> Any:
        """The flight recorder — one row per act of thinking, kept so a *join* can be read.

        Every organ keeps its own counters and none of them can answer "under which conditions
        does my strategy historically fail?", because that is a question about strategy ×
        situation and the average is the only thing anything held. See :mod:`nyxara.njp.blackbox`.
        """
        if not self._gate("blackbox", True):
            return None
        try:
            from nyxara.njp.blackbox import BlackBox
            return BlackBox()
        except Exception:  # noqa: BLE001 — she thinks the same, she just cannot review it
            return None

    def _build_evolution(self, c: Any) -> Any:
        """Phase 7 — structural change to her own cognition, adopted only on evidence.

        A knob is :meth:`nyxara.njp.field.RecursiveCognitiveField.meta_cycle`'s business. This one
        owns what a knob cannot reach: a new operator, a new representation, a new strategy, a new
        edge of the selection graph — each through sandbox, benchmark, adversarial, regression and
        an old-versus-new comparison before anything is promoted. See :mod:`nyxara.njp.evolution`.
        """
        if not self._gate("evolution", True):
            return None
        try:
            from nyxara.njp.evolution import CognitiveEvolution
            return CognitiveEvolution(gates=bool(self._cfg("evolution_gates", True)))
        except Exception:  # noqa: BLE001 — she keeps the cognition she was written with
            return None

    def _build_doing(self, c: Any) -> Any:
        """Stage G: the actions she can actually take, and choosing between them.

        Not a body. A set of cognitive actions she already performs on fixed cadences, plus the
        one thing nothing did — asking *which of them would move the number that is stuck*. See
        :mod:`nyxara.njp.doing`.
        """
        if not self._gate("doing", True):
            return None
        try:
            from nyxara.njp.doing import CognitiveAgency
            return CognitiveAgency()
        except Exception:  # noqa: BLE001 — she keeps acting on the cadence and choosing nothing
            return None

    def _build_index(self, c: Any) -> Any:
        """§27: the one number she is not allowed to compute about herself.

        `njp/index.py` has computed the eight-term vector since it was written and **nothing has
        ever called it** — `measure()` had zero callers in the package, so the engine the plan
        names as the answer to "capability claim nahi — evidence" produced evidence for nobody.
        """
        if not self._gate("index", True):
            return None
        try:
            from nyxara.njp.index import IntelligenceIndex
            # Width 6, which is the benchmark's own default and not a speed compromise. Four
            # was tried and it reads `G = 0.000` — the generalization stage teaches six members
            # and holds four out, so a narrower run does not measure it faster, it measures
            # something else and reports the answer as a zero. A term silently zeroed by a
            # config choice takes the whole product with it: `I_t` read 0.0000 while every other
            # term was healthy.
            return IntelligenceIndex(seed=self._cfg("index_seed", 0),
                                     width=self._cfg("index_width", 6))
        except Exception:  # noqa: BLE001 — she is measured by her organs' counters alone
            return None

    def _build_society(self, c: Any) -> Any:
        """§19: eight specialists over one world model, in sequence — never a vote.

        Three of the eight existed before this and none over NJP's world model: `Explorer` and
        `Skeptic` are classes in `growth/ecosystem.py`, `Scientist` one in `growth/scientist.py`,
        and Mathematician, Engineer, Historian, Strategist and Judge did not exist at all.
        """
        if not self._gate("society", True):
            return None
        try:
            from nyxara.njp.society import CognitiveSociety
            return CognitiveSociety()
        except Exception:  # noqa: BLE001 — every organ still works alone, as it always has
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
            # Registered as fallbacks. `LearningLoop._install_repairs` owns these kinds whenever
            # the loop is on, and these are what remains when it is off — which is a supported
            # configuration and was, until this, an undocumented behaviour difference decided by
            # construction order.
            for kind, repair in ((ErrorKind.GROUNDING, self._repair_grounding),
                                 (ErrorKind.WORLD_MODEL, self._repair_world),
                                 (ErrorKind.MEMORY, self._repair_memory),
                                 (ErrorKind.RELATION, self._repair_relation),
                                 (ErrorKind.REASONING, self._repair_reasoning)):
                engine.register_repair(kind, repair, owner="brain", default=True)

            # The plan did what it predicted would work and it did not work. The action it took is
            # the thing that was wrong, and `agency._credit` moves exactly that action's record —
            # one arm, not the model. This kind had no repair at all: it was diagnosed, counted,
            # and dropped.
            engine.register_repair(ErrorKind.PLANNING, self._repair_planning, owner="brain")

            # And the one that honestly has no organ. The content was right and the rendering
            # lost it; rendering belongs to `njp.voice`, which has a faithfulness *check* and no
            # learning surface of any kind — no counters, nothing that adapts. Inventing a repair
            # to fill the row would be worse than saying so.
            engine.mark_unrepairable(
                ErrorKind.LANGUAGE,
                "rendering is njp.voice's, and it has a faithfulness check but nothing that learns")
            return engine
        except Exception:  # noqa: BLE001 — she still learns per-organ, just without the diagnosis
            return None

    def _slot_claims(self, thought: NJPThought) -> Tuple[str, str, Dict[str, str]]:
        """What each seat says fills this turn's ``(subject, predicate)`` slot.

        The one place a word and a sentence are commensurable, and the reason the seats could
        never be compared before. ``fabric_proposes`` returns a bare concept *name*;
        ``grounding.Answer`` is a clause; the comparison ran ``_norm`` over both, so AGREE was
        effectively unreachable and every pairing landed in UNCLASSIFIED. A channel that always
        disagrees carries exactly as little information as one that always agrees.

        The slot is borrowed from what :meth:`nyxara.njp.integrate.LearningLoop._open_deferred`
        already keys a deferred question on, rather than parsed a second time here.
        """
        empty: Tuple[str, str, Dict[str, str]] = ("", "", {})
        try:
            if self.grounder is None:
                return empty
            reader = getattr(self.grounder, "_read_question", None)
            if reader is None:
                return empty
            subject, predicate = reader(str(thought.stimulus or "").lower())
            if not subject or not predicate:
                return empty

            fillers: Dict[str, str] = {}
            said = str(thought.answer or "").strip()
            if said:
                # The last token of a short answer is the filler; a long one is a sentence rather
                # than a slot claim and is left out instead of being guessed at.
                words = said.split()
                if len(words) <= 3:
                    fillers["njp"] = words[-1].strip(".,;:")
            if self._gate("fabric_prior", True):
                for name, _score in self.fabric_proposes(thought.stimulus, k=1):
                    if name:
                        fillers["fabric"] = str(name)
            return subject, predicate, fillers
        except Exception:  # noqa: BLE001 — an unreadable slot is simply not compared
            return empty

    def _substrate_disagreement(self, thought: NJPThought) -> Optional[str]:
        """Compare the seats on one slot, and open a question where they differ.

        Two invariants, and they are the Master's own point cutting both ways:

        * **Agreement raises nothing.** Two seats saying the same thing is mostly evidence about
          their shared inputs. :meth:`~nyxara.njp.grounding.Provenance.weakest` is already the
          precedent — a composition is only as sourced as its weakest part.
        * **Disagreement never refutes.** It lowers confidence and opens an experiment. The
          fabric associates; it does not derive, and it is measured at MRR 0.238. A channel that
          wrong must not be able to retract a grounded claim.
        """
        try:
            subject, predicate, fillers = self._slot_claims(thought)
            if len(set(fillers.values())) < 2:
                return None            # agreement, or only one seat spoke. Neither is a clash.
            if self.epistemic is not None:
                rivals = self.epistemic.rivals_for_slot(subject, predicate, fillers)
                if rivals:
                    self.epistemic.compile(thought.stimulus, rivals)
            thought.epistemic_confidence = min(
                float(thought.epistemic_confidence or 0.0), _SUBSTRATE_CEILING)
            from nyxara.njp.router import DisagreementKind
            return DisagreementKind.SUBSTRATE
        except Exception:  # noqa: BLE001
            return None

    # ---- promoting a reasoning form into a strategy ------------------------ #
    def _shape_strategy(self, shape: Tuple[str, ...]) -> Any:
        """Bind one reasoning form into something :mod:`nyxara.njp.metareason` can choose.

        The callable is the walk with the predicate sequence *pinned*, which is the whole content
        of naming a shape: the search that found this chain the first forty times does not have to
        run the forty-first. Everything else about it is an ordinary arm — it is scored by the
        same bandit, on the same outcomes, and a promoted shape that stops working loses its
        record like any other.
        """
        def solve(problem: str, ctx: Dict[str, Any]) -> Any:
            try:
                if self.learner is None:
                    return None
                subject = str(ctx.get("subject") or "").strip()
                if not subject and self.grounder is not None:
                    reader = getattr(self.grounder, "_read_question", None)
                    if reader is not None:
                        subject = str(reader(str(problem or "").lower())[0] or "")
                if not subject:
                    return None
                derived = self.learner.walk_shape(subject, shape)
                return derived.answer or None if derived.ok else None
            except Exception:  # noqa: BLE001
                return None
        return solve

    def _promote_shapes(self) -> int:
        """Register every reasoning form the genome says is worth naming. Returns how many are new.

        :meth:`~nyxara.njp.genome.ReasoningGenome.candidates` has computed this for a long time —
        enough real uses to be a pattern, a success rate that says following it is a good idea, and
        a positive MDL saving so naming it costs less than re-deriving it — and had no consumer
        anywhere. A shape she had re-derived forty times appeared in a stats dict and was
        re-derived a forty-first.

        ``liabilities()`` is never promoted, and that is not the same as merely not being a
        candidate: a form that recurs constantly and is usually *wrong* would, if named, teach her
        to make the same mistake faster.
        """
        if self.genome is None or self.metareason is None:
            return 0
        added = 0
        try:
            from nyxara.njp.metareason import ProblemKind
            liabilities = {tuple(s.shape) for s in self.genome.liabilities()}
            for candidate in self.genome.candidates():
                shape = tuple(candidate.shape)
                if not shape or shape in liabilities:
                    continue
                if len(set(shape)) == 1:
                    # A shape of one predicate repeated is transitivity, and `_strategy_compose`
                    # already owns that walk at *any* length. Promoting it registers a second arm
                    # that does the same thing at a **pinned** length, on the record the first arm
                    # earned — so it outranks `compose` and then returns nothing the moment the
                    # chain is one hop longer than the form that was promoted.
                    #
                    # Found by `njp.school`'s `depth` subject, and it looked like the teaching had
                    # failed: after distillation `core.connects` derived a four-hop chain at 0.18
                    # and `_strategy_compose` returned "yes" when called directly, while `think`
                    # answered nothing and named `shape:p>p>p` as the arm it had used. Teaching had
                    # worked perfectly; what it also did was mint a rival that could not do the
                    # job. `core.walk_shape` says which of the two owns this: "a shape as
                    # `njp.genome` records it is a sequence of *different* predicates".
                    continue
                name = "shape:" + ">".join(shape)
                if name in self.metareason.strategies:
                    continue
                self.metareason.register(
                    name, (ProblemKind.FACTUAL, ProblemKind.CAUSAL),
                    self._shape_strategy(shape),
                    # Its own measured success rate, not a number anyone picked. A shape promoted
                    # on a strong record starts believed to that degree and no further.
                    prior=float(candidate.success if candidate.success is not None else 0.5))
                self._remember_skill(
                    name, f"reason by {' → '.join(shape)}",
                    detail=(f"used {candidate.seen}× with {candidate.success:.0%} success, "
                            f"saving {candidate.saving:.2f}"))
                added += 1
        except Exception:  # noqa: BLE001 — a failed promotion leaves the registry as it was
            return added
        return added

    def _remember_skill(self, key: str, text: str, *, detail: str = "") -> None:
        """File a skill in **procedural** memory — the level that holds *how to do something*.

        Phase 5 names procedural memory as a thing to build, and measured over a full session the
        level was empty: ``{'working': 0, 'episodic': 42, 'semantic': 8, 'procedural': 0}``. Not
        because nothing qualified — she was inventing reusable programs and promoting reasoning
        shapes on their own cadences — but because neither had anywhere to be *kept*. A skill that
        lives only in the organ that produced it is one the rest of her cannot find.

        Procedural is the right level rather than a convenient one: its policy is stability by
        **use** over ninety days, where episodic decays and semantic is promoted by recurrence.
        A way of doing something is not a fact that keeps being restated; it is a capability that
        should survive not being needed for a while.
        """
        if self.levels is None or not key:
            return
        try:
            from nyxara.njp.levels import Level
            self.levels.remember(key, f"{text} — {detail}" if detail else text,
                                 level=Level.PROCEDURAL, source="skill", claim=key)
        except Exception:  # noqa: BLE001 — an unfiled skill still works where it was made
            return

    # ---- the repairs, one per error kind ---------------------------------- #
    def _repair_planning(self, outcome: Any) -> bool:
        """A plan reached the wrong state: charge the action it took, and only that action.

        `ActionValue` is a tried/worked pair per action, and `agency._credit` is the one call that
        moves it. That makes this the same shape of surgery as `_repair_relation` — one arm of one
        organ, chosen by what the failure names — rather than a scalar fed back into everything,
        which is the generic repair the whole error taxonomy exists to avoid.

        An outcome that does not name an action is not repaired. Guessing which action to blame
        would train against a plan she may never have run.
        """
        try:
            agent = getattr(self, "agent", None)
            if agent is None:
                return False
            context = getattr(outcome, "context", None) or {}
            action = str(context.get("action") or "").strip()
            if not action:
                return False
            agent._credit(action, worked=False)
            return True
        except Exception:  # noqa: BLE001
            return False

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

    def _repair_relation(self, outcome: Any) -> bool:
        """A value she composed turned out wrong: make **that** inheritance worth less.

        The targeted repair, and the reason :attr:`~nyxara.njp.predict.ErrorKind.RELATION` was
        worth adding. She answered ``water`` for what a sparrow needs by carrying a property of
        birds down to a member, the Master said ``food``, and the thing that failed is neither her
        perception nor her memory nor her facts — every premise was true. It is the defeasible
        step, for **this predicate**, being more general than the world.

        So exactly one number moves: ``requires`` loses transitivity, and ``is_a``, ``causes`` and
        every other relation are untouched. That is what makes it a repair rather than a retrain —
        the alternative on offer was widening a similarity threshold or writing a ledger line, and
        neither of those makes the next inheritance over the same relation any less confident.

        **Refuted, not zeroed.** :class:`~nyxara.njp.core.Transitivity` is a Beta posterior, so one
        exception moves it by an amount that shrinks as evidence accumulates: a relation that has
        chained correctly a hundred times is barely touched by a single counterexample, and one
        that has never been tested moves a long way. Setting it to zero on a first exception would
        be treating one sparrow as proof that kinds do not carry properties at all.
        """
        try:
            if self.learner is None:
                return False
            context = getattr(outcome, "context", None) or {}
            predicate = str(context.get("predicate") or "").strip()
            if not predicate:
                return False
            self.learner._transitivity(predicate).refute(1.0)
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

    def _build_compiler(self, c: Any) -> Any:
        """Language into a typed operation. See :mod:`nyxara.njp.compile`."""
        if not self._gate("compiler", True):
            return None
        try:
            from nyxara.njp.compile import ThoughtCompiler
            return ThoughtCompiler()
        except Exception:  # noqa: BLE001 — without it interventions simply are not compiled
            return None

    def _build_universe(self, c: Any) -> Any:
        """The internal simulation universe. Takes the world model as its causal skeleton."""
        if not self._gate("universe", True):
            return None
        try:
            from nyxara.njp.universe import InternalUniverse
            universe = InternalUniverse(world=self.world,
                                        max_depth=self._cfg("universe_depth", 6))
            # Import the skeleton the world model already holds, rather than only holding a
            # reference to it. `InternalUniverse(world=...)` uses the world for `_permitted` —
            # which arrows *may* be fitted — and nothing else, so a brain-built universe knew
            # every stated law was permissible and had not declared a single one. `sync_from_world`
            # is the call that turns "this arrow is allowed" into "this arrow exists, and runs
            # this way", and the only two places it was called from were the field loop and a
            # demo. A stated law with no quantities attached is exactly what a text corpus gives
            # her, and until this it reached the simulator as nothing at all.
            try:
                universe.sync_from_world()
            except Exception:  # noqa: BLE001 — an empty skeleton is not a broken universe
                pass
            return universe
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

    def _build_mathematician(self, c: Any) -> Any:
        """Mathematics beyond a closed expression. Absent before this: there was none.

        :meth:`_build_calculator` closed the arithmetic gap and said in its own docstring that it
        was *only* a calculator. Measured on twenty-five ordinary school questions through
        :meth:`think`, with every organ built: **3 right, 17 silent, and 5 filed as facts about
        the world** — reproducible by setting ``mathematics_enabled = False``. See :mod:`nyxara.njp.mathematics`, which owns both halves of that number.
        """
        if not self._gate("mathematics", True):
            return None
        try:
            from nyxara.njp.mathematics import Mathematician
            return Mathematician(calculator=self.calculator,
                                 prefer_exact=self._cfg("calculate_prefer_exact", True))
        except Exception:  # noqa: BLE001
            return None

    def _build_solver(self, c: Any) -> Any:
        """Problems she has never seen. Absent before this: there was nothing that *solved*.

        The mathematician scores 410/410 on its own examination and that number says almost
        nothing — every item on it is a shape it already knows. On thirty problems written to match
        no skill it scored **1 right, 9 confidently wrong, 18 silent**. See
        :mod:`nyxara.njp.mathsolver`.
        """
        if not self._gate("mathsolver", True):
            return None
        try:
            from nyxara.njp.mathsolver import Solver
            return Solver()
        except Exception:  # noqa: BLE001
            return None

    def _build_cognition(self, c: Any) -> Any:
        """The thirteen task shapes of an examination she did not write.

        Measured cold against the sealed split of that corpus, before this existed: **1 right in
        200**, and 34 of the first 97 items filed into the knowledge store as facts about the
        world. See :mod:`nyxara.njp.corpussolver`.
        """
        if not self._gate("corpussolver", True):
            return None
        try:
            from nyxara.njp.corpussolver import CorpusSolver
            return CorpusSolver()
        except Exception:  # noqa: BLE001
            return None

    def _build_cortex(self, c: Any) -> Any:
        """Her cortex, as a proposer. Absent weights make it honestly unavailable, not absent."""
        if not self._gate("cortex", True):
            return None
        try:
            from nyxara.njp.cortex import Cortex
            return Cortex(provider=self._cfg("cortex_provider", "qwythos"),
                          max_tokens=self._cfg("cortex_max_tokens", 768))
        except Exception:  # noqa: BLE001
            return None

    def _build_router(self, c: Any) -> Any:
        """The seat router. Given the classifier, the self-model and the bandit rather than its own
        copies: a router that classified for itself would put a keyword list in competition with a
        strategy selector that learns from outcomes, and the keyword list would win arguments it
        should lose.

        ``self.meta`` is the :class:`~nyxara.njp.selfmodel.MetaLearner` this brain already built —
        the same one ``metareason`` uses. One learner, one record of what actually works; its arms
        are namespaced (``strategy:<kind>`` there, ``seat:<kind>`` here) so they do not collide.
        """
        if not self._gate("router", True):
            return None
        try:
            from nyxara.njp.router import Router
            return Router(meta=self.metareason, self_model=self.self_model, learner=self.meta)
        except Exception:  # noqa: BLE001
            return None

    def _build_embedding(self, c: Any) -> Any:
        """Entity similarity over the manifold's own vectors. Off leaves every query unwidened."""
        if not self._gate("embedding", True):
            return None
        try:
            from nyxara.njp.embed import EntityEmbedding
            return EntityEmbedding(self.fabric.manifold, cell_id=_cell_id,
                                   floor=self._cfg("embedding_floor", 0.35))
        except Exception:  # noqa: BLE001
            return None

    def _build_competition(self, c: Any) -> Any:
        """Rank rival accounts. Off leaves the router's two-seat reconciliation exactly as it was."""
        if not self._gate("competition", True):
            return None
        try:
            from nyxara.njp.compete import Competition
            return Competition(world=self.world, self_model=self.self_model,
                               floor=self._cfg("competition_floor", 0.35))
        except Exception:  # noqa: BLE001
            return None

    def _build_epistemic(self, c: Any) -> Any:
        """The compiler that turns an unresolved question into the observation that would settle
        it. Given curiosity because that is what already retires a question once it is answered."""
        if not self._gate("epistemic", True):
            return None
        try:
            from nyxara.njp.epistemic import EpistemicCompiler
            return EpistemicCompiler(universe=self.universe, curiosity=self.curiosity,
                                     stakes=self._cfg("epistemic_stakes", 0.5))
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
            # `self_model` is built at line 346, before this — so the gate has its record from
            # the first turn rather than being attached afterwards and missing the early ones.
            meta = MetaReasoner(meta_learner=self.meta, beliefs=self.beliefs,
                                world=self.world, universe=self.universe,
                                self_model=self.self_model)
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
                # Composition along one relation, which `derive` does not do: `derive` chains
                # *different* predicates (`is_a` then `needs`), and this chains one with itself.
                # Two different walks, and only the first of them had a caller.
                # **FACTUAL only, and the reason is arithmetic.** `max_attempts` is 3, and CAUSAL
                # already has `causal`, `simulate` and `derive`. A fourth arm there does not add
                # an option, it makes one unreachable: `test_a_strategy_that_produces_nothing_
                # yields_to_one_that_can` exists because `causal` abstains on an intervention and
                # `simulate` has to be reachable behind it, and a fourth candidate pushed
                # `simulate` past the budget. Causal reachability is answered by `_strategy_causal`
                # instead, which is the arm that owns causal questions.
                meta.register("compose", (ProblemKind.FACTUAL,),
                              self._strategy_compose, prior=0.55)
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

    def _build_noesis(self, c: Any) -> Any:
        """The library of reasoning she has compressed into reusable primitives.

        Attached because :mod:`nyxara.njp.index` needs it and could not read it: ``R`` — reasoning
        that survived a held-out description-length win — was permanently absent from the vector
        with the honest note *"no noesis engine attached"*, while the engine sat fully built one
        package over. A term whose source exists and is simply not wired is the same class of gap
        as an organ that reports zeros.

        Nothing here drives it. It is constructed and held, so the index can read its curve and a
        caller can step it; the WAKE/SLEEP/DREAM loop stays something run deliberately rather than
        something every turn pays for.

        **The transfer verifier is attached, and Phase 5's milestone is why.** That milestone is
        two claims — ``skills_created > 0`` *and the skills transfer to unseen tasks* — and the
        engine has always had the hook for the second (``self.transfer.transfer_all(...)``, called
        on every SLEEP) with nothing in the repository implementing it. The attribute defaulted to
        ``None``, the guard above the call was always false, and ``transferred`` was structurally
        zero on every cycle ever run. See :mod:`nyxara.growth.zeroshot`.
        """
        if not self._gate("noesis", True):
            return None
        try:
            from nyxara.growth.noesis import NoesisEngine
            from nyxara.growth.zeroshot import ZeroShotTransfer
            return NoesisEngine(seed=self._cfg("seed", 42),
                                tasks_per_cycle=self._cfg("noesis_tasks_per_cycle", 12),
                                transfer=ZeroShotTransfer())
        except Exception:  # noqa: BLE001 — a brain without a program library still thinks
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

    def _build_coder(self, c: Any) -> Any:
        """The organ that reads a program, runs it, writes one from examples, and repairs one.

        Everything else in this brain produces claims that carry a confidence and go to the
        gauntlet. Code is the one thing here with a decision procedure attached — it runs on the
        examples or it does not — so :mod:`nyxara.njp.coding` needs none of the organs above it
        and none of them need to weigh what it returns.
        """
        if not self._gate("coder", True):
            return None
        try:
            from nyxara.njp.coding import Coder
            return Coder(max_attempts=self._cfg("coder_max_attempts", 20000),
                         max_steps=self._cfg("coder_max_steps", 20000),
                         graft=bool(self._cfg("coder_graft", True)))
        except Exception:  # noqa: BLE001 — a brain that cannot program still thinks
            return None

    def _build_discourse(self, c: Any) -> Any:
        """The organ that reads a *conversation* rather than a sentence.

        Empty on day one in the same sense :meth:`_build_language` is: it holds no convention
        until one has been demonstrated, no referent until one has been introduced, and it calls
        nothing figurative until the store has witnessed enough to make that a judgement rather
        than a guess. A brain that is never taught and never talked to behaves exactly as it did
        before this organ existed.

        It is handed :meth:`_kinds_of` — her own ``is_a`` store — as its kinds oracle, so what she
        can tell about a metaphor is exactly what she has been taught about the world.
        """
        if not self._gate("discourse", True):
            return None
        try:
            from nyxara.njp.discourse import Communicator
            return Communicator(kinds=self._kinds_of,
                                speaker=str(self._cfg("discourse_speaker", "Master")),
                                addressee=str(self._cfg("discourse_addressee", "NYXARA")))
        except Exception:  # noqa: BLE001 — a brain with no discourse organ still reads sentences
            return None

    def _build_language(self, c: Any) -> Any:
        """The organ that can acquire a grammar she was not shipped with.

        Empty on day one and that is the honest default: it holds no construction, no affix and
        no word class until somebody has shown it a sentence with its meaning, so a brain that is
        never taught behaves exactly as it did before this organ existed. Nothing drives it — no
        pulse, no evolver — because everything it learns comes from a demonstration, and a
        demonstration is something a caller makes rather than something a clock produces.
        """
        if not self._gate("language", True):
            return None
        try:
            from nyxara.njp.language import LanguageFaculty
            return LanguageFaculty(default=str(self._cfg("language_default", "en")))
        except Exception:  # noqa: BLE001 — a brain with no grammar organ still reads English
            return None

    # ---- language --------------------------------------------------------- #
    def hear_language(self, text: str, *, tongue: Optional[str] = None) -> None:
        """Overhear a sentence: no meaning attached, and nothing asserted from it.

        This feeds the two organs that need only exposure — the affix induction and the word
        classes — and it feeds neither the grammar nor the fact store. Hearing a sentence is not
        being told a fact, and a faculty that treated it as one would be inventing beliefs out of
        overheard strings.
        """
        if self.language is not None:
            self.language.hear(text, tongue=tongue)

    def show_language(self, surface: str, meaning: Any, *,
                      tongue: Optional[str] = None) -> bool:
        """One demonstration: a sentence, and the meaning it carries. Generalised by
        :meth:`learn_language` and by nothing before it."""
        if self.language is None:
            return False
        return self.language.show(surface, meaning, tongue=tongue)

    def learn_language(self, *, tongue: Optional[str] = None) -> Any:
        """Generalise the demonstrations into constructions, and report what was thrown away.

        Deliberately not driven, for the reason :meth:`go_to_school` is not: this is the call that
        changes how every later sentence is read, and a thing that changes how she reads is a
        thing a caller should have to ask for by name.
        """
        if self.language is None:
            return None
        return self.language.learn(tongue=tongue)

    def read_language(self, surface: str, *, tongue: Optional[str] = None) -> Any:
        """Parse with the learned grammar alone, returning a
        :class:`~nyxara.njp.semantics.Meaning`. Unreadable is a real answer here."""
        if self.language is None:
            return None
        return self.language.read(surface, tongue=tongue)

    def learn_languages(self, *, courses: Any = None) -> Any:
        """Sit the English and Hindi curriculum and come back with the report card.

        Deliberately not driven, for the reason :meth:`go_to_school` is not: it changes how every
        later sentence is read. It is also the only call in this brain that gives her a
        **vocabulary** — until it runs she knows no content word in any language, which is the
        honest state of a system that ships a grammar and no dictionary.
        """
        try:
            from nyxara.njp.lessons import enrol
            if self.language is None:
                return None
            _faculty, reports = enrol(self.language,
                                      **({"courses": courses} if courses else {}))
            return reports
        except Exception:  # noqa: BLE001
            return None

    def translate(self, surface: str, *, into: str, frm: Optional[str] = None) -> str:
        """Read in one language and say it in another, or say nothing.

        The words cross by a taught glossary and the rest crosses as meaning. One unglossed word
        and the answer is the empty string rather than a sentence with a hole in it.
        """
        if self.language is None:
            return ""
        return self.language.translate(surface, into=into, frm=frm)

    def say_language(self, meaning: Any, *, tongue: Optional[str] = None) -> str:
        """Put a meaning into words, or return nothing.

        Every sentence this produces has been parsed again and checked to give back the meaning it
        started from. That is the condition :mod:`nyxara.njp.voice` has always insisted on before
        she is allowed a fluent surface, met mechanically instead of by an apology.
        """
        if self.language is None:
            return ""
        return self.language.say(meaning, tongue=tongue)

    # ---- discourse -------------------------------------------------------- #
    def hear_turn(self, text: str, *, speaker: Optional[str] = None) -> Any:
        """Read one turn as part of a **conversation** and return the whole uptake.

        The counterpart to :meth:`read_language`, one level out: that returns what a sentence
        means, and this returns what somebody was doing by saying it, what its pronouns name, and
        what it does to the claims already on the ledger. ``think`` calls this for every turn — it
        is exposed separately so a caller can inspect the reading without spending a whole turn.
        """
        try:
            if self.discourse is None:
                from nyxara.njp.discourse import Uptake
                return Uptake(surface=str(text or ""))
            return self.discourse.hear(str(text or ""), speaker=speaker)
        except Exception:  # noqa: BLE001
            from nyxara.njp.discourse import Uptake
            return Uptake(surface=str(text or ""))

    def show_act(self, surface: str, act: str) -> None:
        """Demonstrate that this surface does this — see
        :meth:`~nyxara.njp.discourse.ActLearner.show`. Two demonstrations with different fillers
        are the minimum that leaves anything behind."""
        try:
            if self.discourse is not None:
                self.discourse.show(str(surface or ""), str(act or ""))
        except Exception:  # noqa: BLE001
            return

    def correct_act(self, surface: str, *, took_as: str, meant: str) -> bool:
        """A communication that failed, handed back as a lesson. Returns whether the correction
        **generalised** rather than merely fixing its own sentence."""
        try:
            if self.discourse is None:
                return False
            return bool(self.discourse.misread(str(surface or ""), took_as=str(took_as),
                                               meant=str(meant)))
        except Exception:  # noqa: BLE001
            return False

    def expect_turn(self) -> Any:
        """What she expects of the next turn, before it arrives — see
        :class:`~nyxara.njp.discourse.Anticipation`. An organ that has heard nothing returns an
        empty expectation, which is a different answer from a confident wrong one."""
        try:
            if self.discourse is None:
                from nyxara.njp.discourse import Expectation
                return Expectation()
            return self.discourse.anticipation.expect()
        except Exception:  # noqa: BLE001
            from nyxara.njp.discourse import Expectation
            return Expectation()

    def show_change(self, first: str, second: str, verdict: str) -> None:
        """Demonstrate what a speaker would call this pair of claims — ``updates``,
        ``contradicts`` or ``corroborates``. What word carried the verdict is induced rather than
        stated; see :class:`~nyxara.njp.discourse.MarkerLearner`."""
        try:
            if self.discourse is not None:
                self.discourse.show_change(str(first or ""), str(second or ""), str(verdict or ""))
        except Exception:  # noqa: BLE001
            return

    def show_alternation(self, plain: str, marked: str) -> None:
        """Demonstrate that these two sentences mean the same thing, one of them in a shape the
        compiler cannot read — a passive, an auxiliary chain, a gapped coordinate. Where each
        slot's filler ends up is induced from the pair rather than written down."""
        try:
            if self.discourse is not None:
                self.discourse.show_alternation(str(plain or ""), str(marked or ""))
        except Exception:  # noqa: BLE001
            return

    def show_exchange(self, first: str, second: str, relation: str) -> None:
        """Demonstrate what the second turn is doing to the first — answering it, accepting it,
        clarifying it. The **acts** are what is recorded, never the sentences, so one
        demonstration of a question being answered reads any question being answered."""
        try:
            if self.discourse is not None:
                self.discourse.show_exchange(str(first or ""), str(second or ""),
                                             str(relation or ""))
        except Exception:  # noqa: BLE001
            return

    def show_attachment(self, surface: str, site: str) -> None:
        """Demonstrate which site a prepositional phrase attached to — ``event`` or ``object``.
        A preposition demonstrated both ways is one this language uses both ways."""
        try:
            if self.discourse is not None:
                self.discourse.show_attachment(str(surface or ""), str(site or ""))
        except Exception:  # noqa: BLE001
            return

    def fit_reference(self, cases: Any) -> Any:
        """Move the pronoun resolver's cue weights to whatever answers these demonstrations
        best — see :meth:`~nyxara.njp.discourse.Reference.fit`."""
        try:
            if self.discourse is None:
                return {}
            return self.discourse.fit_reference(cases)
        except Exception:  # noqa: BLE001
            return {}

    def say_to(self, meaning: Any, *, audience: str = "student") -> Any:
        """One meaning said for one hearer, parsed back before it is returned — see
        :class:`~nyxara.njp.discourse.Register`."""
        try:
            if self.discourse is None:
                from nyxara.njp.discourse import Said
                return Said(audience=audience)
            return self.discourse.say(meaning, audience)
        except Exception:  # noqa: BLE001
            from nyxara.njp.discourse import Said
            return Said(audience=audience)

    def holds(self, subject: str, relation: str) -> str:
        """What the conversation currently says about this, or ``""`` where it says two things.

        The empty string on a contested pair is the point of the ledger and is not the same answer
        as "nothing was said": a store that returned the later claim would be the silent overwrite
        :mod:`nyxara.njp.discourse` exists to refuse.
        """
        try:
            if self.discourse is None:
                return ""
            return self.discourse.holds(str(subject or ""), str(relation or ""))
        except Exception:  # noqa: BLE001
            return ""

    def go_to_discourse_school(self, *, rounds: int = 2, seed: int = 26,
                               subjects: Any = None) -> Any:
        """Sit the syllabus about being talked to: acts, reference, contradiction, minds,
        register, and the two subjects that are controls on the other nine."""
        try:
            from nyxara.njp.discourseschool import DiscourseSchool
            return DiscourseSchool(seed=seed, rounds=rounds, subjects=subjects).attend(self)
        except Exception:  # noqa: BLE001
            return None

    # ---- programming ------------------------------------------------------ #
    def read_code(self, source: str) -> Any:
        """Read a small piece of Python into a program she can run and explain.

        Raises :class:`~nyxara.njp.coding.CodeError` on anything outside the reader's whitelist,
        and that refusal is the security boundary rather than a limitation to be worked around:
        what comes back is a term tree, and the only machine that runs a term tree is her own
        interpreter.
        """
        from nyxara.njp.coding import read_python
        if self.coder is None:
            raise RuntimeError("the coding faculty is switched off")
        return read_python(source)

    def run_code(self, source: str, *args: Any) -> Any:
        r"""Read it and run it on ``args``. Nothing is ever ``eval``\ ed or ``exec``\ ed."""
        program = self.read_code(source)
        return self.coder.run(program, args)

    def trace_code(self, source: str, *args: Any) -> List[Any]:
        """Every sub-expression with the value it took — the answer to "why does it print that"."""
        program = self.read_code(source)
        return self.coder.trace(program, args)

    def write_code(self, spec: Any, *, attempts: Optional[int] = None) -> Any:
        """Write a program that reproduces every shown example of ``spec``, or abstain.

        There is no best-effort return. A program that nearly works is not a partial answer, and
        the one thing this brain is built never to do is hand back something unverified as though
        it were verified.
        """
        if self.coder is None:
            return None
        return self.coder.write(spec, attempts=attempts)

    def learn_code(self, spec: Any, source: str) -> Any:
        """Take a worked solution, verify it by running it, and keep only its shape.

        The demonstrated answer is discarded. What is retained is the skeleton, which is what
        applies to a task nobody demonstrated — the same rule :mod:`nyxara.njp.teacher` applies
        to a demonstrated chain, on the one kind of claim that can be checked mechanically.
        """
        if self.coder is None:
            return None
        return self.coder.learn_python(spec, source)

    def go_to_school(self, *, rounds: int = 1, seed: int = 7,
                     subjects: Any = None) -> Any:
        """Sit the reasoning and coding syllabus and come back with the report card.

        Deliberately not driven by anything: like :meth:`report_card` this is run when somebody
        asks for it. Teaching mutates her — it states facts, it moves a transitivity posterior,
        and on one subject it raises a search budget — and a thing that mutates her is a thing a
        caller should have to ask for by name.
        """
        try:
            from nyxara.njp.school import School
            return School(seed=seed, rounds=rounds, subjects=subjects).attend(
                self, coder=self.coder)
        except Exception:  # noqa: BLE001
            return None

    def sit_general_exam(self, *, limit: int = 400, seed: int = 20260902,
                         papers: Any = None) -> Any:
        """Sit the general-knowledge examination over whatever world corpus is already in her.

        The counterpart to :meth:`go_to_school`, and the opposite of it in one respect worth
        stating: the school teaches and this does not. Every paper in
        :mod:`nyxara.njp.general` reads through :meth:`Grounder.answer` and
        :meth:`CognitiveLearningCore.predict` and writes nothing, so this may be run twice and
        give the same answer twice — which is what makes it usable as a before-and-after around
        something that *does* mutate her.

        It examines **this** brain rather than loading the shipped corpus into a fresh one, so a
        brain that has never been fed world knowledge scores honestly rather than being quietly
        topped up first.
        """
        try:
            from nyxara.njp.general import GeneralKnowledgeExam
            return GeneralKnowledgeExam(self, limit=limit, seed=seed).sit(papers)
        except Exception:  # noqa: BLE001
            return None

    def puzzle(self, question: str) -> Any:
        """Answer a question whose answer is in no single fact she holds.

        The counterpart to :meth:`Grounder.answer`, and the division between them is the point:
        that one *retrieves*, this one *constructs*. "What is the currency of the country where the
        Taj Mahal is" has no key in the store under any spelling — it is four locating hops and a
        lookup, and :mod:`nyxara.njp.puzzle` walks them and reports the path it walked.

        Returns a :class:`~nyxara.njp.puzzle.Solution`, empty where nothing could be constructed.
        Nothing is written: a constructed answer is an inference, and filing it would let the next
        question read it back as though somebody had stated it.
        """
        try:
            from nyxara.njp.puzzle import PuzzleSolver
            return PuzzleSolver(self).solve(question)
        except Exception:  # noqa: BLE001
            return None

    def do_maths(self, question: str) -> Any:
        """Work out a mathematics question and show the working.

        Asks :mod:`nyxara.njp.mathsolver` first and :mod:`nyxara.njp.mathematics` second, which is
        the order :meth:`_mathematical` uses and for the reason given there.

        The in-process entry point to :mod:`nyxara.njp.mathematics`, and the counterpart to
        :meth:`puzzle` in the same way :meth:`puzzle` is the counterpart to ``Grounder.answer``:
        that one *retrieves*, this one *computes*. Returns a
        :class:`~nyxara.njp.mathematics.Solution` carrying the topic, the method, the steps and
        whether the value is exact — ``ok`` is False and ``error`` says why for anything it
        declines, which is most sentences and is the point.

        Nothing is written. A computed value is not a fact somebody stated, and filing it would
        let the next turn read it back as though it were — the same rule :meth:`puzzle` keeps for
        a constructed answer, for the same reason.
        """
        try:
            if self.solver is not None:
                solved = self.solver.solve(question)
                if solved.ok or solved.recognised:
                    return solved
            if self.mathematician is None:
                from nyxara.njp.mathematics import Solution
                return Solution(question=str(question or ""),
                                error="the mathematics organ is switched off")
            return self.mathematician.solve(question)
        except Exception:  # noqa: BLE001
            return None

    def go_to_maths_school(self, *, rounds: int = 1, seed: int = 11,
                           subjects: Any = None) -> Any:
        """Sit the mathematics syllabus — pre-tested, taught, then examined on unseen items.

        The counterpart to :meth:`go_to_school`, and it teaches for the same reason that one
        does: what a lesson can move here is not the arithmetic — that is a decision procedure and
        was right before the lesson — but *what she can be asked*, and the formulas she can state
        back when nobody is asking her to compute anything.
        """
        try:
            from nyxara.njp.mathschool import MathSchool
            return MathSchool(seed=seed, rounds=rounds, subjects=subjects).attend(self)
        except Exception:  # noqa: BLE001
            return None

    def sit_maths_exam(self, *, limit: int = 40, seed: int = 20260902,
                       papers: Any = None) -> Any:
        """Sit the mathematics examination: every topic, generated items, nothing taught.

        Writes nothing and teaches nothing, so it may be run twice around something that mutates
        her and the two numbers are comparable — the discipline :meth:`sit_general_exam` keeps,
        for the same reason.
        """
        try:
            from nyxara.njp.mathschool import MathExam
            return MathExam(self, limit=limit, seed=seed).sit(papers)
        except Exception:  # noqa: BLE001
            return None

    def do_cognitive(self, task: str) -> Any:
        """Read one item from an outside cognitive corpus and solve it, with the working.

        The counterpart to :meth:`do_maths`, and read-only in the same sense: a solved chain is
        not a fact somebody stated, so nothing reaches the store. An item no reading claims comes
        back ``recognised=False`` rather than with a guess — and one that *is* claimed and
        declined comes back ``recognised=True, ok=False``, which is a different answer and is kept
        as one.
        """
        try:
            if self.cognition is None:
                from nyxara.njp.corpussolver import Reading
                return Reading()
            return self.cognition.solve(task)
        except Exception:  # noqa: BLE001
            from nyxara.njp.corpussolver import Reading
            return Reading()

    def go_to_corpus_school(self, *, rounds: int = 1, seed: int = 25,
                            subjects: Any = None) -> Any:
        """Sit the syllabus built around a corpus nobody in this repository wrote.

        The counterpart to :meth:`go_to_maths_school`, and the difference is the only thing it is
        for: that school's questions and its code have the same author, and this one's do not.
        """
        try:
            from nyxara.njp.corpusschool import CorpusSchool
            return CorpusSchool(seed=seed, rounds=rounds, subjects=subjects).attend(self)
        except Exception:  # noqa: BLE001
            return None

    def sit_corpus_exam(self, *, split: str = "EVAL", limit: Optional[int] = 200,
                        seed: int = 20260902) -> Any:
        """Sit the sealed split of that corpus. Writes nothing and teaches nothing.

        Answers live in a file this path never opens; the grader is the corpus's own. Reported by
        faculty, by generator and by verifier type, because a single aggregate hides exactly the
        thing the split policy was built to expose.
        """
        try:
            from nyxara.njp.corpusschool import Examination
            return Examination(self, split=split, limit=limit, seed=seed).sit()
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

    #: What a polar answer looks like when some other path has already reduced it to one.
    #: Registers rather than synonyms: she answers in both, and a verdict in either is still a
    #: verdict, not an object to be checked against what was asked.
    _POLAR_VERDICTS_LOCAL = ("yes", "no", "haan", "nahi", "haan.", "nahi.")

    @staticmethod
    def _settle_polar(thought: NJPThought) -> None:
        """Turn a derived *object* into the yes/no a polar question actually asked for.

        Deliberation and recall answer a ``(subject, relation)`` pair with whatever object they
        can reach, and they have no idea the question already named one. Two failures follow from
        that, and they need opposite remedies — which is why this compares rather than blocks:

        * Taught only ``X needs A`` and asked ``"do Xs need B?"``, she replied ``A``. True,
          confident, and an answer to a question nobody asked. The derived object does not match,
          so there is nothing to affirm and the reply is silence.
        * Told ``aag causes garmi`` and ``garmi causes pasina`` and asked
          ``"does aag cause pasina?"``, the answer is **yes** — and it is only reachable through
          the two-hop composition :mod:`nyxara.njp.core` performs. An earlier version of this
          guard stopped deliberation outright on a polar gap and lost exactly that: the grounder's
          own single ``is_a`` hop cannot compose a causal chain, and refusing to let anything else
          try made the polar path strictly weaker than the wh-path over the same knowledge.

        So the ladder runs in full and its result is *checked* here. Matching is on the canonical
        form at both ends, because "pasina" derived and "pasina" asked are one answer however each
        was spelled.
        """
        try:
            grounding = getattr(getattr(thought, "percept", None), "grounding", None)
            answer = getattr(grounding, "answer", None)
            if not getattr(answer, "polar", False):
                return
            asked = str(getattr(answer, "polar_object", "") or "").strip().lower()
            derived = str(thought.answer or "").strip().lower()
            if not asked or not derived:
                return
            # Already a verdict. The ladder has its own polar handling on some paths — a two-hop
            # causal chain comes back as "yes" rather than as the object it walked to — and this
            # method is here to convert *objects*, not to re-judge judgements. Without the guard
            # it compared "yes" against the asked object, found no match, and discarded a
            # correctly composed answer: the measured cost was
            # ``test_a_composed_answer_is_reachable_from_a_spoken_question``.
            if derived in NJPBrain._POLAR_VERDICTS_LOCAL:
                return
            # Substring in either direction: a derivation legitimately arrives with its route
            # attached ("pasina, via garmi"), and the asked object is legitimately a shorter
            # phrase than the stored one.
            if asked in derived or derived in asked:
                thought.answer = "yes"
                return
            # Derived something else entirely. The pair was never established, and naming the
            # other object would be answering a different question.
            thought.answer = ""
        except Exception:  # noqa: BLE001
            return

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
                self._recall_knowledge(thought)
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

    def _recall_knowledge(self, thought: NJPThought) -> None:
        """Last resort: passages the kernel's KnowledgeBase holds, when the fact store had nothing.

        This exists because the handle was dangling. ``attach_kernel`` has always set
        ``self.knowledge``, and across the whole of ``nyxara/njp/`` it was never read once — so
        everything the kernel ingested, and everything :mod:`nyxara.growth.foraging` distilled
        from the live web, went into a store her brain could not reach. She foraged the internet
        and did not know she had.

        Three things keep it from becoming a hole in the epistemics:

        * **It runs only when the fact store came back empty.** A retrieved passage never
          outranks a stated relation.
        * **It is never ``KNOWN``, and it never enters the fact store.** A chunk of text that
          mentions what was asked is evidence of *relevance*, not of truth — the same distinction
          ``answer_by_recall`` already refuses to blur — and passing it to ``_assert`` would let
          prose she has not parsed become a fact she would later state.
        * **It says where it came from.** ``recalled`` names the source, so a reply built on a
          document is distinguishable in the trace from one built on something she was told.
        """
        try:
            if self.knowledge is None:
                return
            chunks = self.knowledge.retrieve(thought.stimulus, k=1) or []
            if not chunks:
                return
            text = str(getattr(chunks[0], "text", "") or "").strip()
            if not text:
                return
            thought.answer = text[:1000]
            thought.recalled = f"knowledge base: {getattr(chunks[0], 'source', '?')}"
        except Exception:  # noqa: BLE001 — a missing knowledge base is absent, not fatal
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
            # "does aag cause pasina" is a causal question about *two named things*, and
            # explanation does not answer it — `world.why("aag")` explains the wrong end. The
            # chain that does answer it is `core.connects`, priced by the predicate's own
            # transitivity. Handled here rather than by a fourth CAUSAL arm; see the registration.
            composed = self._strategy_compose(problem, ctx)
            if composed:
                return composed
            subject = str(ctx.get("subject") or "").strip().lower()
            if not subject or self.world is None:
                return None
            why = self.world.why(subject)
            if not getattr(why, "explained", False):
                return None
            return ", ".join(c.cause for c in getattr(why, "causes", [])[:3])
        except Exception:  # noqa: BLE001
            return None

    def _strategy_compose(self, problem: str, ctx: Dict[str, Any]) -> Any:
        """Walk the relation the question names, however many hops it takes.

        **The machinery has been here the whole time and nothing could reach it.**
        :meth:`~nyxara.njp.core.CognitiveLearningCore.reach` and
        :meth:`~nyxara.njp.core.CognitiveLearningCore.connects` compose *any* predicate, priced by
        that predicate's own transitivity posterior — and the only two calls to either, anywhere in
        the package, pass the literal string ``"causes"`` from inside ``core.py`` itself. The
        seven-stage benchmark's own comment records the same finding from the other side: it used
        to call them directly, "so it measured the *organ* and reported 20/20 while the same
        question asked in words returned ''".

        So a causal chain composed through ``think`` and every other relation did not. Measured::

            zorbo se vanth hoti hai / vanth se pluron hoti hai
            "zorbo se kya kya hota hai"     -> "pluron"     ✓
            zorbo kizzles vanth / vanth kizzles pluron
            "does zorbo kizzle pluron?"     -> ""           ✗   (core.connects: yes, 0.234)

        The parse was never the problem either: :func:`~nyxara.njp.semantics.compile_meaning`
        returns ``subject='zorbo', relation='kizzle', object='pluron'`` for that sentence. Three
        finished parts, no wire between them.

        **One hop is not this strategy's answer.** ``recall`` owns lookups, and returning a direct
        edge here would take credit for them — the bandit would then learn to prefer this arm for
        questions it adds nothing to. Only a genuinely *composed* derivation is returned.
        """
        try:
            if self.learner is None:
                return None
            from nyxara.njp.semantics import compile_meaning
            meaning = compile_meaning(problem)
            relation = str(getattr(meaning, "relation", "") or "").strip().lower()
            subject = str(getattr(meaning, "subject", "") or "").strip().lower()
            obj = str(getattr(meaning, "object", "") or "").strip().lower()
            if not relation or not subject:
                return None
            # "what does zorbo *ultimately* kizzle" parses its subject as "zorbo ultimately" —
            # an adverb sits inside the noun phrase and the tagger has no word class for it.
            # Resolved against the store rather than by a stop-list: the prefix that actually has
            # outgoing edges of this relation is the subject, and a name nothing knows anything
            # about is not one.
            for candidate in self._subject_prefixes(subject, relation):
                resolved = self._resolve_relation(candidate, relation)
                got = (self.learner.connects(candidate, obj, resolved) if obj
                       else self.learner.reach(candidate, resolved))
                if getattr(got, "kind", "") != "composed":
                    continue
                # Kept for the caller, which prices the answer's confidence off the derivation
                # that produced it. See `_answer_by_reasoning`.
                self._composition = got
                return "yes" if obj else str(getattr(got, "answer", "") or "") or None
            return None
        except Exception:  # noqa: BLE001
            return None

    def _resolve_relation(self, subject: str, relation: str) -> str:
        """The predicate the *store* filed, given the verb the question used.

        They are not the same string and cannot be assumed to be. "does aag cause pasina" compiles
        to ``cause``; the grounder files ``causes``. It happened to match for ``kizzle`` and that
        near-miss is exactly how this kind of bug survives: the first case tried works, so nothing
        looks wrong until a relation whose lemma differs comes along and the walk silently finds
        no edges at all.

        Resolved against the subject's own outgoing predicates rather than by a lemmatiser, so it
        can only ever return a relation this subject actually has.
        """
        try:
            wanted = relation.rstrip("s")
            predicates = {str(p) for s, p, _o, _c in self.learner._edges()
                          if str(s) == subject}
            if relation in predicates:
                return relation
            for predicate in sorted(predicates):
                if predicate.rstrip("s") == wanted:
                    return predicate
        except Exception:  # noqa: BLE001
            return relation
        return relation

    def _subject_prefixes(self, subject: str, relation: str) -> List[str]:
        """``subject`` and the shorter readings of it that the store has heard of."""
        tokens = subject.split()
        out = [subject]
        for size in range(len(tokens) - 1, 0, -1):
            shorter = " ".join(tokens[:size])
            if shorter and shorter not in out:
                out.append(shorter)
        return out[:3]

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
            direction = int(ctx.get("direction", 0) or 0)
            out = self.universe.intervene({variable: value},
                                          stated_direction={variable: direction})
            if not out.answerable:
                return None
            effects = [d for d in out.deltas
                       if d.variable != variable
                       and (d.qualitative or (d.change is not None and abs(d.change) > 1e-6))]
            if not effects:
                return None

            was = out.factual.get(variable)
            if was is not None:
                done = f"{variable} {float(was):g} → {value:g}"
            elif direction:
                # No quantity was ever observed for this variable, so there is no "6 → 3" to
                # state. The sentence still said which way, and saying that is not the same as
                # inventing a number for it.
                done = f"{variable} {'lower' if direction < 0 else 'higher'}"
            else:
                done = f"{variable} = {value:g}"

            said: List[str] = []
            for delta in effects[:3]:
                if delta.qualitative or delta.after is None or delta.before is None:
                    # Direction only. Rendered as a direction, never as a number she does not
                    # have — the old rendering formatted `before`/`after` unconditionally and
                    # raised on exactly this case, so a qualitative answer became no answer.
                    if delta.direction > 0:
                        said.append(f"{delta.variable} higher")
                    elif delta.direction < 0:
                        said.append(f"{delta.variable} lower")
                    else:
                        said.append(f"{delta.variable} changes, direction unknown")
                else:
                    said.append(f"{delta.variable} {delta.before:g} → {delta.after:g}")
            entails = "; ".join(said)
            note = f" — {out.reason}" if out.reason else ""
            return f"{done} predicts {entails} (confidence {out.confidence:.2f}){note}"
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
            # A variable she has an *arrow* for but has never put a number to. Measured: over a
            # session where every stated law reached the skeleton, `state` was `{}` and stayed
            # `{}`, because it is filled only from concepts carrying numbers and a text corpus
            # states laws without quantities. Resolving against observed values alone therefore
            # rejected every variable she actually knew about, and "paani" — sitting in three
            # relations — came back unresolvable.
            #
            # Second, and only second: a measured variable is the better answer whenever there is
            # one, because it can be intervened on quantitatively.
            for (cause, effect) in list(getattr(self.universe, "relations", None) or {}):
                for variable in (cause, effect):
                    if variable == want or variable.rsplit(".", 1)[-1] == want:
                        return variable
        except Exception:  # noqa: BLE001 — an unresolvable name simply is not simulated
            return ""
        return ""

    def _counterfactual_context(self, question: str, *, intent: Any = None,
                                act: Any = None) -> Dict[str, Any]:
        """The intervention a question asks for, compiled rather than pattern-matched.

        This used to be four bare regexes joined by ``or``, and it is the whole reason
        :mod:`nyxara.njp.compile` exists: ``or`` short-circuits, the English word-order pattern
        matched a Hinglish sentence and took the verb as the variable, and the pattern written for
        that word order never ran. Without a ``variable`` key :meth:`_strategy_simulate` returns
        ``None`` on every call, so the do-operator was registered, chosen, run, and unable to do
        anything — and ``metareason._miss`` charged each of those losses to ``simulate``.

        A factor still resolves against what was actually observed, so "halve the water" means
        half of what this plant has been getting rather than half of an arbitrary unit. What is
        new is that a sentence stating a direction and no quantity now compiles too: *"paani aadha
        kar doon"* is a decrease whether or not anything has ever been measured.
        """
        if self.universe is None or self.compiler is None:
            return {}
        try:
            # Laws stated earlier in this session reach the simulator here. The field loop also
            # syncs, but it runs at step 11 — after this — so without it a law stated on turn N
            # was unusable until turn N+1, which reads from outside as her having forgotten it.
            try:
                self.universe.sync_from_world()
            except Exception:  # noqa: BLE001
                pass
            op = self.compiler.compile(
                str(question or ""), intent=intent, act=act,
                resolve=self._resolve_variable,
                state=getattr(self.universe, "state", None) or {})
            return op.to_context()
        except Exception:  # noqa: BLE001 — an uncompiled intervention is simply not simulated
            return {}

    def _closed_arithmetic(self, problem: str) -> bool:
        """Is this a question the calculator can *close*? Then no guess may answer it.

        Found by :mod:`nyxara.njp.school`, and worth stating as the rule rather than as the fix:
        asked ``"what is 25 + 10?"`` after a few similar turns she answered **10**. Every organ
        behaved: the deliberation ladder settled on a token the question itself contained, marked
        the conclusion decided, and the meta-reasoner — which learns its ordering from outcomes,
        so a prior does not decide it — took that ahead of ``calculate``. A recalled token is a
        plausible answer to an arithmetic question in exactly the way that is worst: it is a
        number, it is in the right range, and it is wrong.

        The rule the repo already holds is *verifiable beats probabilistic*, and this is that rule
        applied one level earlier. A question with a decision procedure does not get a guess
        offered against it at all — not weighed against it, not ranked below it, not offered.
        ``_strategy_calculate`` closes the expression or returns ``None``, so when this reads true
        there is an arm that cannot be wrong, and every other arm abstaining costs nothing.
        """
        try:
            if self.calculator is None:
                return False
            return bool(self.calculator.evaluate(problem).ok)
        except Exception:  # noqa: BLE001
            return False

    def _mathematical(self, problem: str) -> Any:
        """The mathematician's reading of this turn — a ``Solution`` when a skill claimed it.

        **Claimed, not solved, and the difference is a defect the exam found.** "convert 5 km to
        kilograms" is recognised by the conversion skill, which refuses it because length and mass
        are different quantities — the refusal that skill exists to make. Gating on *solved*, the
        turn then fell through to the grounder as an ordinary statement and was filed:
        ``('convert', '5') → 'km kilograms'``. That is the original defect surviving inside its
        own fix, on the one input where the fix declines. A recognised task is a task whether or
        not it has an answer, and neither kind is a claim about the world.

        Two callers, and they are the two halves of :mod:`nyxara.njp.mathematics`'s reason for
        existing. :meth:`perceive` reads a non-``None`` as *do not file this as a fact*;
        :meth:`_compose` answers from it when it is also ``ok``, which is the rule
        :meth:`_closed_arithmetic` already states for a closed expression applied to everything
        else that has a decision procedure.

        **Bare arithmetic is deliberately excluded.** A closed expression already has a route:
        the calculator is registered as a strategy, the classifier is fed a parsed-expression flag
        so the critic does not discard the value as an untested empirical claim, and
        :meth:`_closed_arithmetic` stops the ladder guessing over it. That machinery was built
        against a measured failure and it works. Answering ``2+2`` here instead would leave every
        piece of it in the source and unreachable — dead wiring created on purpose. So the
        mathematician owns what nothing owned, and arithmetic stays where it already worked.

        The memo inside :meth:`Mathematician.solve` makes asking twice cost once.
        """
        try:
            # **The solver is asked first, and the order is the whole of its value.** A reading
            # that carries several constraints is checked against all of them before she may
            # speak; a skill that matches one phrase is not checked against anything. Asked the
            # other way round, "marks up by 40% then discounts 25%" is answered **30** by the
            # discount skill — a percentage it recognised, in a problem it did not — and the
            # solver never gets a turn. Verified beats matched, exactly as verifiable beats
            # probabilistic one layer up.
            if self.solver is not None:
                solved = self.solver.solve(problem)
                # `recognised` blocks as firmly as `ok` answers, and that is the point of it:
                # a problem the solver understood and declined must not be handed down to the
                # skill table, which understands it less and answers anyway. Measured: "the sum to
                # infinity of 2, 4, 8, …" — a series with no sum — came back **14**, the total of
                # the three terms the skill table could see.
                if solved.ok or solved.recognised:
                    return solved
            if self.mathematician is None:
                return None
            worked = self.mathematician.solve(problem)
            if worked.ok:
                return None if worked.topic == "arithmetic" else worked
            # A refusal *with a topic* is a skill declining a question it understood. A refusal
            # without one is the dispatcher saying no skill recognised the sentence at all, and
            # that sentence may well be an ordinary statement the grounder should learn from.
            return worked if worked.topic else None
        except Exception:  # noqa: BLE001
            return None

    def _cognitive(self, problem: str) -> Any:
        """This turn's reading by :mod:`nyxara.njp.corpussolver`, or ``None``.

        Asked **before** :meth:`_mathematical` for the reason that method gives about its own
        solver, one level further out: these readings are the most specific in the package. Each
        needs a whole structure to be present — ``Start with x = …`` *and* numbered steps *and* a
        modulus; a rule base *and* a named goal — so a reading that claims a turn has already
        established far more about it than any phrase match could, and nothing it claims is
        something the mathematician wanted. A test pins that: the twenty-eight-paper maths
        examination reports the same score with this organ present as without it.
        """
        try:
            if self.cognition is None:
                return None
            reading = self.cognition.solve(problem)
            return reading if reading.recognised else None
        except Exception:  # noqa: BLE001
            return None

    def _is_cognitive_task(self, problem: str) -> bool:
        """The store question, asked for thirteen more shapes.

        :meth:`_is_maths_task` was written against exactly this failure and covers arithmetic
        only, so an outside corpus walked straight past it: *"Move the seal from the brown sack to
        the black tin."* is a well-formed statement about the world, and the grounder was filing
        it as one — 34 times in the first 97 items of the sealed split, at the same confidence as
        a fact she had been taught.

        A recognised task is a task **whether or not it has an answer**, which is why this asks
        ``recognised`` and not ``ok``: the five constraint puzzles she declines as ambiguous are
        no more claims about the world than the ones she solves.
        """
        try:
            if self.cognition is None:
                return False
            return bool(self.cognition.recognised_task(problem))
        except Exception:  # noqa: BLE001
            return False

    def _is_maths_task(self, problem: str) -> bool:
        """Is this an instruction to work something out — *even though nothing could*?

        A narrower question than :meth:`_mathematical` and used for one narrower purpose: the
        store. A problem she cannot solve is still not a fact about the world, and without this the
        grounder filed the ones that beat her — ``('find', 'the') → 'smallest positive integer n'``
        among them, at the same confidence as a stated fact.

        It never decides who answers. Conflating it with a refusal cost a working capability
        once already: "expand (x+2)(x+3)" is a task the solver has no reading for and the skill
        table expands correctly, and a task flag that blocked turned that into silence.
        """
        try:
            if self.solver is None:
                return False
            return bool(self.solver.solve(problem).task)
        except Exception:  # noqa: BLE001
            return False

    def _strategy_ladder(self, problem: str, ctx: Dict[str, Any]) -> Any:
        try:
            if self.reasoner is None:
                return None
            if self._closed_arithmetic(problem):
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
                answer = str(cached)
            else:
                derived = self.learner.answer(problem)
                if not derived.ok:
                    return None
                answer = derived.answer or None
            if not answer:
                return None
            # **A polar question names the object it is asking about.** `core.answer` returns the
            # store's best object for the subject whatever was asked, so "does zorbo kizzle
            # pluron?" came back "vanth" — non-empty, criticised clean, and accepted. The turn was
            # then answered, so the retry loop never ran and `compose` — which composes the two
            # hops the question actually asks about — never got a turn. Answering a different
            # question is not a defect the critic can see, because the answer it is handed is a
            # perfectly good answer to a question nobody asked.
            from nyxara.njp.semantics import compile_meaning
            meaning = compile_meaning(problem)
            asked = str(getattr(meaning, "object", "") or "").strip().lower()
            if str(getattr(meaning, "focus", "") or "") == "truth" and asked:
                if str(answer).strip().lower() != asked:
                    return None
                return "yes"
            return answer
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
        """Words to cells. Stable across restarts, which is what makes the fabric reloadable.

        Every call also records the inverse. :func:`_cell_id` is a blake2b digest and is therefore
        one-way by construction — there is no arithmetic that recovers ``"sparrow"`` from the
        integer it hashes to. That is fine for addressing a cell and it is exactly what stopped the
        substrate from ever saying anything: :meth:`~nyxara.njp.learn.Readout.predict` returns
        gradient-trained probabilities over *cell ids*, which no reader and no other seat can do
        anything with.

        The map is not hard to recover, though, because she computes it herself on every turn and
        was throwing it away. Keeping it is the whole of the fix.
        """
        out: List[int] = []
        for token in self.concepts(text):
            cell = _cell_id(token)
            # First spelling wins. Two labels colliding on one 48-bit digest is vanishingly
            # unlikely, and if it ever happens the honest thing is to keep naming the cell what it
            # was first called rather than to let a later word silently rename an established one.
            self.lexicon.setdefault(cell, token)
            out.append(cell)
        return out

    def fabric_proposes(self, text: str, *, k: int = 5,
                        floor: float = 0.0) -> List[Tuple[str, float]]:
        """What the substrate expects to go with this turn, **as concepts she can name**.

        The fabric's own account of a turn, produced by the one learner in the package that uses
        real gradients rather than local coincidence. It is deliberately not routed into
        :meth:`_compose`: this is association learned from co-activation, it knows nothing about
        relations or truth, and an answer path that took it would be guessing fluently.

        What it is *for* is being a third account that was produced differently from the other two
        — which is the only thing that makes a disagreement between them informative.
        """
        try:
            if self.readout is None:
                return []
            cells = self.encode(text)
            # Widen by entities whose relational context resembles this one's. This is the whole
            # point of `njp.embed`: the head learned about `sparrow`, and without this `finch`
            # addresses a slot it has never seen a gradient for. Measured on six groups with no
            # `is_a` anywhere, held-out MRR went 0.542 → 1.000 against an unmoved control.
            if self.embedding is not None:
                for concept in self.concepts(text)[:2]:
                    for neighbour in self.embedding.expand(concept):
                        cells = cells + self.encode(neighbour)
            return self.readout.predict_symbols(
                cells, lexicon=self.lexicon, k=k, floor=floor)
        except Exception:  # noqa: BLE001
            return []

    def name_cell(self, cell_id: int) -> str:
        """What this cell is called, or ``""`` when she has never encoded that concept.

        Empty rather than a placeholder: a cell she cannot name is one the readout must decline to
        report, and a stand-in string would be a symbol she never learned appearing in a proposal.
        """
        return self.lexicon.get(int(cell_id), "")

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
                # Heard as a *conversation* before it is read as a sentence. Two things come
                # out of this that the grounder cannot produce for itself: a surface whose
                # settled pronouns have been filled in, and the judgement that a sentence is not
                # meant literally. Both have to happen before extraction, because a fact filed is
                # a fact that outlives the turn that filed it.
                heard = out.stimulus
                if self.discourse is not None:
                    try:
                        out.uptake = self.discourse.hear(out.stimulus)
                        heard = out.uptake.rewritten or out.stimulus
                    except Exception:  # noqa: BLE001
                        out.uptake = None

                if self.grounder is not None:
                    try:
                        # **A sum is not a sentence about the world.** The grounder reads a
                        # non-question as a statement and files what it extracts, and a maths
                        # imperative parses as one: "simplify the fraction 18/24" was asserted as
                        # ('simplify fraction', '18') → '24' at confidence 0.75, in the same store
                        # inheritance and the puzzle solver walk. Asking her to do arithmetic was
                        # writing nonsense into what she reasons from. `store` is False exactly
                        # when the mathematician can close the turn, so nothing is extracted from
                        # a task — see `Grounder.ground`.
                        # And a metaphor is not a sentence about the world either. *"The market
                        # swallowed the shock"* was filed as ``market swallow shock`` at the same
                        # confidence as a fact she had been taught — the V.23 store defect again,
                        # in a shape neither the arithmetic guard nor the cognitive one covers,
                        # because it is a well-formed statement in every respect except being
                        # meant.
                        out.grounding = self.grounder.ground(
                            heard, intent=intent,
                            store=self._mathematical(out.stimulus) is None
                            and not self._is_maths_task(out.stimulus)
                            and not self._is_cognitive_task(out.stimulus)
                            and (out.uptake is None or out.uptake.literal))
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
                        # What is firing right now, handed to recall. It was computed one line
                        # above and thrown away on every turn: the fabric settled, produced its
                        # fired set, and then recall ran on the raw cue string alone. Two memories
                        # whose text looks alike but that were met in entirely different states
                        # are indistinguishable to content addressing, and the thing that tells
                        # them apart was already in scope.
                        #
                        # Gated, because whether it *helps* is a question for the ablation rather
                        # than for anyone's intuition, and it is the one fabric arm that can
                        # change an answer rather than only lower a confidence.
                        fired: Tuple[int, ...] = ()
                        if self._gate("fabric_retrieval", True):
                            fired = tuple(int(c) for c in (out.cells or ()))
                        out.recall = self.memory.recall(out.stimulus,
                                                        k=int(spend["recall_k"]),
                                                        context=fired)
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
            # `field._predict_world` labels its transition with `brain._last_intent_kind`, and
            # **nothing in the package ever assigned it** — one read, no writes, so every
            # observation the field gave the dynamics model carried the empty action label while
            # the loop's half carried a real one. Two halves of one model disagreeing about
            # whether an action happened. Written here, where the intent is first known.
            self._last_intent_kind = str(getattr(out.intent, "kind", "") or "")

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
            # **A procedure that says "there is no answer" is an answer.** The restraint paper in
            # `mathschool` measured what happens without this line: after a lesson stated "a mode
            # is the value that appears most often in a list", the control *"what is the mode of
            # 1, 2, 3, 4, 5?"* — a list with no mode, which the statistics skill correctly
            # declines — was answered with that definition. `_compose` had already returned
            # silence; deliberation and recall ran anyway, because they run whenever the answer is
            # empty, and recall offered the nearest thing in the store.
            #
            # This is `_closed_arithmetic`'s rule at its limit: a question with a decision
            # procedure does not get a guess offered against it, and that has to hold when the
            # procedure's output is "none" as much as when it is a number. Otherwise every honest
            # refusal in `mathematics` becomes an invitation for a plausible substitute.
            # The same rule, for the organ added in V.25, and it had to be written separately
            # because it reads a different field. Measured with this clause absent: the five
            # seating puzzles `corpussolver` declines as genuinely ambiguous came back **"4"** and
            # **"5"** — seat numbers, offered by recall against a question asking for an item.
            # Five abstentions turned into five confident errors, which is the worst trade in this
            # package: precision fell and nothing was gained. A procedure that says "there is more
            # than one answer" is an answer, exactly as "there is none" is.
            refused = ((out.mathematics is not None
                        and not getattr(out.mathematics, "ok", False))
                       or (out.cognition is not None
                           and not getattr(out.cognition, "ok", False)))
            if not out.answer and not refused:
                # A prediction that resolves **inside this turn**, which is the whole test the
                # raw-stimulus key failed: she commits to whether the ladder will find an answer,
                # and microseconds later it either did or did not. Nothing has to arrive later and
                # nothing waits in the open queue.
                #
                # It is also the one prediction the self-model can act on directly — knowing when
                # deliberating is worth the descent is a capability, separately measurable from
                # deliberating well, and `outcome` routes the miss to `reasoning` so the record
                # that `metareason._demoted` reads is the record this loop produced.
                deliberation = self._predict_deliberation(out)
                self._deliberate(out)
                self._score_deliberation(out, deliberation)
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
                # A polar question's derived object becomes the yes/no it asked for, before the
                # epistemic pass reads the answer — otherwise the state is computed for a reply
                # that is about to change.
                self._settle_polar(out)
                if out.answer:
                    self._set_epistemic(out)
                    # A deliberated answer came by a *form* of reasoning, and the genome has been
                    # keeping that form's record. Applied here rather than inside
                    # `_set_epistemic` because only this branch produced one: a grounded answer
                    # is a lookup with no shape to hold responsible.
                    out.epistemic_confidence = self._temper_by_shape(
                        out.epistemic_confidence, out)
                else:
                    self._ask_back(out)
            # Sure of herself on ground the substrate does not recognise. Asked after the
            # confidence is settled, because both halves of the condition have to be known, and it
            # produces a question rather than touching the answer.
            self._wonder_at_anomaly(out)
            # And: the two substrates naming different fillers for one slot. Also after the
            # confidence is settled, because what it does is cap it.
            out.substrate = self._substrate_disagreement(out)

            # 4.5 CONSULT THE CORTEX — the one step that makes a stronger model buy stronger
            # reasoning rather than better sentences. It proposes; the gate below still decides.
            self._consult_cortex(out)

            # 4.6 COMPETE — rank every account of this turn rather than counting them. On the
            # main path deliberately: the arbitration above needs a cortex and there usually is
            # not one, and a ranking that only runs when a language model is attached is the
            # same defect `epistemic` had — an organ reachable in principle and never in fact.
            # NJP and the fabric are two accounts, which is a field.
            self._compete(out)

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
            self._observe_contexts(out)

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

            # 10b. RECORD — one row for this act of thinking, after the loop so the report is
            # on the thought and before the field so the row describes the turn as it was
            # answered rather than as the field left it.
            if self.blackbox is not None:
                self.blackbox.record(out)

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
                    # And, on the same slow count and behind its own gate, the riskiest thing
                    # here: giving up a causal model that keeps being surprised. Off by default,
                    # because a wrong death criterion destroys working structure and
                    # `concepts.restructure` records a measured instance of exactly that. The
                    # trial itself is reversible — archived, benchmarked on held-out, reverted on
                    # a tie — but "reversible" is a reason to allow it to be switched on, not a
                    # reason to switch it on for everyone.
                    if self._gate("model_death", False):
                        out.retirement = self.field.retire_cycle()

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

    def _observe_contexts(self, thought: NJPThought) -> None:
        """Give this turn's relations to the entity embedding, as (predicate, object) contexts.

        Read off the *store* rather than the turn, so an entity's vector is the bundle of
        everything she holds about it rather than of whatever the last sentence happened to
        mention. A representation rebuilt from one turn's fragment would move an entity every time
        it was talked about, and similarity would track recency instead of structure.
        """
        try:
            if self.embedding is None or self.grounder is None:
                return
            grounding = getattr(thought.percept, "grounding", None)
            for triple in list(getattr(grounding, "triples", None) or [])[:8]:
                subject = str(getattr(triple, "subject", "") or "")
                if not subject:
                    continue
                contexts = [(str(t.predicate), str(t.object))
                            for t in self.grounder._facts_of(subject)
                            if getattr(t, "predicate", "") and getattr(t, "object", "")]
                if contexts:
                    self.embedding.observe(subject, contexts)
        except Exception:  # noqa: BLE001
            return

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
                # And the same episode under the subject's **kind**, which is the one thing that
                # makes any of this compress.
                #
                # Every antecedent set here is built from the *instance*, so `sparrow requires
                # water`, `crow requires water` and `robin requires water` share no subset at all
                # and every candidate rule has support 1. Measured over a whole session: 70
                # episodes, 4 discovery passes, **0 abstractions**, compression 1.0 — which
                # `concepts` defines as "does not pay for itself". The compressor was not failing;
                # it was being fed data in which nothing could recur.
                #
                # Recorded under the kind the Master **stated**, never one she guessed, for the
                # same reason `field._kind_of` prefers a stated kind: an inferred kind would make
                # a rule out of her own conjecture and then test it as though it came from the
                # world. The generalisation is not trusted for being proposed — the discoverer
                # fits on one split and scores on held-out episodes, so a kind-level rule that
                # does not hold dies exactly like any other.
                if obj:
                    for kind in self._kinds_of(subject):
                        # Substituting a subject with its kind turns `sparrow is_a bird` into
                        # `{bird, is_a} → bird` — a tautology, and one that scores five supports
                        # and crowds out the rules worth finding. A rule whose antecedent already
                        # contains its consequent says nothing about the world.
                        if kind == consequent:
                            continue
                        self.discoverer.observe([kind, predicate], consequent)
            if not seen and concepts and thought.answer:
                self.discoverer.observe(concepts, thought.answer[:60])
        except Exception:  # noqa: BLE001 — a missed episode costs one sample, never the turn
            pass

    def _kinds_of(self, subject: str) -> List[str]:
        """The kinds this subject was *stated* to belong to — one hop, stated only, at most two.

        One hop rather than the full ``is_a`` chain: a walk to the root would file the same
        episode under `sparrow`, `bird`, `animal` and `thing`, and a rule about `thing` is a rule
        about everything, which compresses nothing and pollutes every held-out score with a
        pattern that trivially holds.
        """
        try:
            grounder = self.grounder
            if grounder is None or not subject:
                return []
            out: List[str] = []
            for triple in grounder._lookup(subject, "is_a")[:2]:
                kind = str(getattr(triple, "object", "") or "").strip().lower()
                if kind and kind != subject:
                    out.append(kind)
            return out
        except Exception:  # noqa: BLE001
            return []

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
                compiled = self._counterfactual_context(
                    thought.stimulus, intent=thought.intent, act=thought.act)
                thought.op = compiled.get("op")
                context.update(compiled)
                # The speech act already decided what kind of question this is, with a margin.
                # Letting `ProblemClassifier` re-derive it from a bag of words is two classifiers
                # with no shared channel, and on the reported sentence they disagreed: the act
                # reader said causal_query at 0.85, the classifier said empirical at 0.50, and the
                # three strategies empirical selects cannot answer an intervention.
                if thought.op is not None and getattr(thought.op, "kind", ""):
                    context["kind_hint"] = thought.op.kind
                # What this speech act permits at all. `CognitivePolicy` has held the right table
                # since it was written and it was read only to decide whether the cortex could
                # speak; strategy selection never asked. So the only thing keeping `simulate` away
                # from a greeting was the problem kind coming out differently, which is a
                # coincidence rather than a rule.
                if thought.op is not None and getattr(thought.op, "pathways", ()):
                    context["pathways"] = tuple(thought.op.pathways)
                # How unrecognised the substrate found this turn. Gated: whether spending a second
                # strategy on unfamiliar ground pays for itself is a question for the ablation.
                # Default OFF, on measurement rather than preference — see `NJPConfig`. Novelty
                # does not discriminate among the turns that reach here.
                if self._gate("fabric_strategy", False):
                    context["novelty"] = float(
                        getattr(thought.percept, "novelty", 0.0) or 0.0)
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
                        # Keep the *shape* of the reasoning, not just its answer. Until now the
                        # derivation survived one turn and was dropped, so two identical chains
                        # through different subjects were never seen together and nothing could
                        # notice she was re-deriving the same form over and over.
                        if self.genome is not None:
                            thought.trace = self.genome.record(
                                derived, question=thought.stimulus)
                # `solve` copies the context, so a strategy cannot hand anything back through it.
                # `compose` walks a chain the pre-computed `learner.answer` above did not — that
                # one stops at the first hop — so without this the turn is answered by one
                # derivation and its confidence is priced off another, or off none at all.
                # Measured: "what does zorbo ultimately kizzle" answering "pluron" and reported as
                # `believed 0.00`, which is not a hedge, it is a contradiction.
                self._composition = None
                solution = self.metareason.solve(thought.stimulus, context=context)
                thought.solution = solution
                if (str(getattr(solution, "strategy", "")) == "compose"
                        and self._composition is not None):
                    thought.derivation = self._composition
                    if self.genome is not None:
                        thought.trace = self.genome.record(self._composition,
                                                           question=thought.stimulus)
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

    def _predict_deliberation(self, thought: NJPThought) -> str:
        """Commit to whether the ladder will answer, before it runs. Returns the key, or empty.

        The prior comes from her own record rather than a constant: a brain that has found
        deliberation fruitful expects it to be, and one that has not, does not. An untested record
        sits at 0.5 and predicts "answered", which is the optimistic default that at least gets the
        loop scored on its first turns.
        """
        predictor = getattr(self, "predictor", None)
        if predictor is None:
            return ""
        try:
            level = 0.5
            model = getattr(self, "self_model", None)
            if model is not None:
                level = float(model.level("reasoning:deliberation"))
            key = f"{getattr(thought, 'cycle_id', '')}:deliberation"
            predictor.predict(key, "answered" if level >= 0.5 else "unanswered",
                              confidence=abs(level - 0.5) * 2.0, organ="reasoning",
                              stimulus=str(getattr(thought, "stimulus", "")))
            return key
        except Exception:  # noqa: BLE001 — the prediction loop is an optional organ
            return ""

    def _score_deliberation(self, thought: NJPThought, key: str) -> None:
        """Reality, one call later. The half that makes the line above a prediction at all."""
        if not key:
            return
        try:
            actual = "answered" if str(getattr(thought, "answer", "") or "").strip() \
                else "unanswered"
            self.predictor.observe(key, actual, evidence={"organ": "reasoning", "triples": True,
                                                          "facts_correct": True})
            model = getattr(self, "self_model", None)
            if model is not None:
                model.observe("reasoning:deliberation", 1.0 if actual == "answered" else 0.0)
        except Exception:  # noqa: BLE001
            pass

    def _turn_salience(self, thought: NJPThought) -> float:
        """How much of a shock this turn was, 0…1 — the number that decides what outlives it.

        The same rule `attention` uses, read from the same place rather than defined twice: the
        prediction loop's `Outcome.surprise` for this cycle when there is one, otherwise the
        manifold's settling residual. Two definitions of "how surprising was that" drifting apart
        is precisely the defect this phase set out to close, and writing a second one here to save
        an attribute lookup would have reopened it in the same commit.

        Without this call the `salience` parameter is decoration: a hook every organ can read and
        nothing ever fills.
        """
        try:
            attention = getattr(self, "attention", None)
            if attention is not None and hasattr(attention, "_scored_surprise"):
                scored, _source = attention._scored_surprise(thought)
                if scored > 0.0:
                    return max(0.0, min(1.0, float(scored)))
            settled = getattr(getattr(thought, "percept", None), "settled", None)
            return max(0.0, min(1.0, float(getattr(settled, "surprise", 0.0) or 0.0)))
        except Exception:  # noqa: BLE001 — an unmeasurable turn is simply not a surprising one
            return 0.0

    def _remember_turn(self, thought: NJPThought) -> None:
        """File this turn's conclusion at the level it belongs to.

        The level is chosen from what the turn was *about*, not from how it was phrased. Anything
        grounded to the Master or to NYXARA herself is autobiographical and therefore protected —
        that is the level whose loss would change who she is, so it is not subject to forgetting
        at all. Everything else is an episode, and becomes semantic only if it recurs.
        """
        try:
            salience = self._turn_salience(thought)
            if self.levels is not None:
                self.levels.remember(
                    f"turn-{self.turns}", thought.answer,
                    level=self._level_for(thought), cue=thought.stimulus,
                    source=f"turn-{self.turns}", claim=self._claim_of(thought),
                    salience=salience,
                    cells=tuple(getattr(thought.percept, "cells", ()) or ()))
                return
            if self.memory is not None:
                self.memory.remember(f"turn-{self.turns}", thought.answer,
                                     kind=("conclusion" if thought.verified else "episode"),
                                     cue=thought.stimulus, salience=salience,
                                     # The state she was in when she learned this. Without it
                                     # stored there is nothing for the recall side to compare a
                                     # later state against.
                                     cells=tuple(getattr(thought.percept, "cells", ()) or ()))
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

            # 0.5. WORKED OUT, not looked up and not deliberated.
            #
            # Ahead of every retrieval below and ahead of the ladder, for the reason
            # `_closed_arithmetic` gives for arithmetic and which is not special to arithmetic: a
            # turn with a decision procedure has an arm that cannot be wrong, and every other arm
            # abstaining costs nothing. It is also the only stage an *imperative* reaches — "expand
            # (x+2)(x+3)" is not a question, so `_deliberate` never sees it and the strategy table
            # is unreachable from it. Putting the mathematician here rather than in that table is
            # what makes one path serve both the question form and the task form.
            #
            # After the social gate above, never before it: "how are you" is not improved by being
            # checked for arithmetic first, and the policy that owns that decision is the policy.
            # Before the mathematician, and a recognised refusal blocks exactly as firmly as an
            # answer does — the rule V.24 established. A constraint puzzle with two consistent
            # answers must not be handed down to something that will find one of them.
            read = self._cognitive(thought.stimulus)
            if read is not None:
                thought.cognition = read
                return str(read.answer)[:2000] if read.ok else ""

            worked = self._mathematical(thought.stimulus)
            if worked is not None:
                thought.mathematics = worked
                if worked.ok:
                    return str(worked.answer)[:1000]
                # Recognised and declined. There is no answer to give and no fact to file, and
                # falling through to recall would offer the nearest thing in the store to a
                # question that has none — which is how "what is 7 divided by 0" came back 0.7.
                return ""

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

            # 1.5. Two claims that cannot both hold, or a pronoun nothing in the discourse
            # settles. Either way she has a question rather than an answer.
            #
            # **Here** rather than lower down, and the position is the whole of it. Below this
            # sits the abstention branch for a turn the grounder called a question — and it calls
            # *"When I visited Delhi last year I was tired."* a question, on the fronted ``when``,
            # so a contradiction she had correctly detected returned the empty string. Above it
            # sits the grounded answer, which still wins: a question structure can actually answer
            # is answered, never asked back.
            #
            # It is also said **instead of** a recalled memory rather than beside one. Reaching
            # into the store on a turn she could not read is exactly how *"He was tired."* came
            # back with the previous turn attached to it at confidence 0.75.
            asked = getattr(getattr(thought, "percept", None), "uptake", None)
            question = str(getattr(asked, "question", "") or "")
            conflict = bool(getattr(getattr(asked, "verdict", None), "conflict", False))
            # And **not** on a turn the grounder called a question, unless the conflict is what
            # this is about. The reason is the one every abstention subject in this repository
            # rests on: a control scores silence as right and *any* non-empty reply as wrong, so a
            # clarification offered where she would have abstained is a regression on every such
            # paper however much better it reads. A contradiction is the exception because it is
            # the case the grounder misreads — *"When I visited Delhi last year…"* is called a
            # question on its fronted `when` — and because a conflict is a fact about what she was
            # told rather than a guess at what was asked.
            if question and (conflict or not getattr(grounding, "is_question", False)):
                said = list(getattr(grounding, "triples", None) or [])
                confirmed = ("noted: " + "; ".join(_confirm(t) for t in said[:3])) if said else ""
                return " — ".join(x for x in (confirmed, question) if x)[:1000]

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
                # 2a. The *conversation* may answer what the store cannot. A location that was
                # stated three turns ago and revised two turns ago is not in the fact store under
                # a relation any question frame asks for — `"the key is in the drawer"` compiles
                # to a bare clause whose verb is `drawer` — and it is on the ledger under
                # ``is_at``, which is where the revision was applied.
                #
                # Narrow on purpose: a ``where`` question, about a subject the ledger holds, and
                # nothing else. And a contested pair answers with the dispute rather than with
                # either claim, because those are different states and returning one of them as
                # though it were settled is the overwrite this whole organ exists to refuse.
                asked_where = self._answer_from_conversation(thought)
                if asked_where:
                    return asked_where
                # One thing is safe to answer from memory here, and only one: a question she was
                # taught *this exact question*. `recall_cue` is a dict lookup on the normalised
                # question text, so it cannot return something merely near what was asked — which
                # is the whole reason the fuzzy path was refused above.
                #
                # Without it she could not answer the questions she had just studied: measured
                # over 200 taught pairs, 8 answered. That is not caution, it is a hole — being
                # unable to repeat what you were told is a failure of memory, not a virtue of
                # abstention. Held-out questions were never taught, find nothing, and still
                # abstain, which is the correct answer for them.
                taught = self._answer_as_taught(thought)
                if taught:
                    return taught
                return ""

            bits: List[str] = []
            # 3. She was told something and it landed. Confirm what was actually understood, so a
            # misparse is visible to the Master on the turn it happens rather than three turns later.
            #
            # **With its polarity.** This line read `f"{t.subject} {t.predicate} {t.object}"`, and
            # `GroundedTriple.negated` — which the compiler sets correctly and the store keeps
            # correctly — was never in it. So *"I never visited Delhi."* came back
            # `noted: Master visit delhi`: the one window the Master has into what was understood,
            # showing the **opposite of what was said**. That is the `("Zorbins don't", requires,
            # glarn)` defect of V.09 surviving in the only place nobody thought to measure, which
            # is not the parse but the confirmation.
            triples = list(getattr(grounding, "triples", None) or [])
            if triples:
                learned = "; ".join(_confirm(t) for t in triples[:3])
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
            # A worked answer is made of the question's own symbols, and that is what being
            # correct looks like here rather than what echoing looks like. `is_meta_commentary`
            # scores word overlap, so it cannot tell the two apart and it blanked both of these:
            #
            #     "is 91 a prime number?"      → "no, 91 is not a prime number"   overlap 0.83
            #     "factorise x^2 + 5x + 6"     → "(x + 3)(x + 2)"                 overlap 0.67
            #
            # Each was the right answer, each was deleted, and the turn went out silent. The
            # exemption is evidence-based exactly as the `triples` one above it is: a solution
            # with working attached is proof she did something to the question rather than
            # repeating it.
            if getattr(getattr(thought, "mathematics", None), "ok", False):
                return False
            # The same exemption, for the same reason, one organ further out — and it had to be
            # written separately because the evidence is a different object. Every answer
            # `corpussolver` produces is built from the item's own symbols, so the overlap score
            # condemns all of them:
            #
            #     "…final value of x modulo 13…"   → "3"        the digit is in the question
            #     "Which item does Devi own?"      → "card"     the word is in the clues
            #
            # Measured: with the organ wired and this line absent, the sealed split scored
            # **0 right in 1414** through `think` while the organ itself was answering 1395 of
            # them correctly. A right answer deleted on its way out is indistinguishable from
            # never having had one.
            if getattr(getattr(thought, "cognition", None), "ok", False):
                return False
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

    def _answer_from_conversation(self, thought: NJPThought) -> str:
        """A ``where`` question answered from the discourse ledger, or ``""``."""
        try:
            if self.discourse is None:
                return ""
            from nyxara.njp.semantics import tag_tokens

            uptake = getattr(getattr(thought, "percept", None), "uptake", None)
            meaning = getattr(uptake, "meaning", None)
            if meaning is None or meaning.kind != "question" or not meaning.subject:
                return ""
            words = {t.text for t in tag_tokens(thought.stimulus)}
            if not (words & {"where", "kahan", "कहाँ"}):
                return ""
            subject = meaning.subject
            held = self.discourse.holds(subject, "is_at")
            if held:
                return held
            return self.discourse.ledger.dispute(subject, "is_at")
        except Exception:  # noqa: BLE001
            return ""

    def _answer_as_taught(self, thought: NJPThought) -> str:
        """The stored answer to *this exact question*, if she was ever taught one.

        Exact, and that is the entire safety argument. Two different questions normalise to two
        different keys, so this can never hand back a loosely-related memory as though it were an
        answer — the failure that made similarity-based recall unusable here, measured at three
        confident wrong answers for three honest abstentions.

        This is memorisation, and memorisation is the bottom rung rather than a shameful one:
        `eval/intelligence.py` scores it as its own stage, and a brain that cannot repeat what it
        was told an hour ago has a broken memory, not high standards.
        """
        try:
            memory = getattr(self, "memory", None)
            lookup = getattr(memory, "recall_cue", None)
            if not callable(lookup):
                return ""
            trace = lookup(thought.stimulus)
            text = str(getattr(trace, "text", "") or "").strip()
            if not text:
                return ""
            thought.recalled = "taught under this exact question"
            # **Priced by whether the store still agrees, not left at zero.** This path returns
            # before anything attaches evidence, so a question asked twice came back as
            # `believed 0.00` on the second ask while the first read `believed 0.448` — and that
            # is not a hedge, it is a contradiction, and it feeds calibration, the black box and
            # the router. Measured at the base commit, so it is not new.
            #
            # The number is a check rather than a floor: `learner.answer` walks the same facts
            # again, and only an answer the walk still reaches gets its confidence. A memory the
            # store can no longer support keeps today's behaviour, which is the correct outcome
            # for it — a recalled claim that no longer follows should not be stated confidently.
            if self.learner is not None:
                try:
                    derived = self.learner.answer(thought.stimulus)
                    if (derived.ok and str(derived.answer or "").strip().lower()
                            == text.strip().lower()):
                        thought.derivation = derived
                        thought.confidence = max(float(thought.confidence or 0.0),
                                                 float(derived.confidence or 0.0))
                except Exception:  # noqa: BLE001 — an uncheckable memory is priced as it was
                    pass
            return text[:1000]
        except Exception:  # noqa: BLE001
            return ""

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

    def _consult_cortex(self, out: NJPThought) -> None:
        """Ask her cortex for rival explanations, and let the record judge them. Never raises.

        This is step 4.5, and it sits exactly here for a reason: **after** NJP has reached its own
        answer and **before** the Truth-Seeking Gauntlet judges it. After, so the cortex is weighed
        against a conclusion rather than filling a vacuum — a proposal with nothing to be selected
        against can only ever be accepted. Before, so nothing it says can skip the gate.

        Three things it may do, and one it may never:

        * a hypothesis that clashes with the **record** simply loses, and the turn is unchanged;
        * a clash between two ungrounded accounts *lowers* confidence and opens a question;
        * that question is compiled into the observation that would settle it, and filed with
          curiosity, which already retires a question once it is answered.

        What it may never do is raise confidence or change the answer. The cortex is a proposer.
        ``relevance.revise_confidence`` already holds the rule this obeys — depth may lower
        confidence, only independent evidence may raise it — and a 9B model is depth, not evidence.
        """
        if self.router is None:
            return
        try:
            out.routing = self.router.route(out.stimulus, act=out.act)
        except Exception:  # noqa: BLE001 — a failed route leaves the turn exactly as it was
            return
        if self.cortex is None:
            return
        if not (out.routing.cortex_permitted or out.routing.extraction_permitted):
            return
        try:
            if not self.cortex.available():
                return
            from nyxara.njp.cortex import CortexReport
            report = CortexReport()
            hypotheses: List[Any] = []
            if out.routing.cortex_permitted:
                # A question: ask for rivals to select between.
                hypotheses = self.cortex.hypotheses(out.stimulus, context=out.answer,
                                                    report=report)
            else:
                # A statement: this is the World Model Generator's turn. Pulling relations and
                # causal claims out of what the Master just said is grounding work, and it is the
                # half of this module that a question never reaches. One cortex call either way —
                # a 9B model is seconds per turn, and two would be spent on the same sentence.
                self.cortex.world_model(out.stimulus, context=out.answer, report=report)
            self.cortex.offer(report, grounder=self.grounder, world=self.world,
                              attacker=self.adversary, beliefs=self.beliefs)
            out.cortex = report
            if not hypotheses or not out.answer:
                return

            from nyxara.njp.grounding import Answer
            from nyxara.njp.router import DisagreementKind, Verdict
            lead = max(hypotheses, key=lambda h: h.confidence)
            njp_side = Answer(text=out.answer, state=out.epistemic,
                              confidence=out.epistemic_confidence, provenance=out.provenance)
            # The third seat. It reviews rather than competes — see :class:`~nyxara.njp.router.Seat`
            # — and what it reviews is whether the substrate had any purchase on this turn at all.
            fabric_side = getattr(out.percept, "anticipated", None) if self._gate("fabric_seat") else None
            out.arbitration = self.router.arbitrate(out.stimulus, cortex=lead, njp=njp_side,
                                                    fabric=fabric_side)

            # A recognition dissent lowers confidence on a verdict that DID settle, so it has to
            # be applied before the disagreement branch returns — otherwise the one case the
            # fabric seat exists to catch, a confident answer on unrecognised ground, is the one
            # case that never reads its result.
            if out.arbitration.disagreement == DisagreementKind.RECOGNITION:
                out.epistemic_confidence = min(out.epistemic_confidence,
                                               out.arbitration.confidence)

            if out.arbitration.verdict != Verdict.DISAGREEMENT:
                return
            # Two accounts that cannot both be right are grounds for LESS certainty in whichever
            # she was about to give, never more. The gauntlet still runs after this.
            out.epistemic_confidence = min(out.epistemic_confidence, out.arbitration.confidence)
            if self.epistemic is None:
                return
            experiment = self.epistemic.compile(out.arbitration.question, hypotheses)
            if experiment is None:
                return
            out.experiment = experiment
            self.epistemic.to_question(experiment)
        except Exception as exc:  # noqa: BLE001 — the cortex is an organ, never a dependency
            log.debug("cortex consult degraded: %s", exc)

    def _compete(self, thought: NJPThought, *, cortex: Any = None, njp: Any = None) -> None:
        """Rank every account of this turn, including the substrate's, and record the outcome.

        The fabric enters as what it is: :meth:`fabric_proposes` comes from a head trained on
        turn-to-turn concept succession, measured at MRR 0.238, carrying no provenance and no
        derivation. It therefore scores ~0 on the two heaviest axes and loses — with the axis
        that cost it recorded. That is the competition working rather than a reason to leave it
        out: ranking is what makes entering a weak account safe, where a vote would have made it
        dangerous.

        Only ever lowers what she will say. On an unsupported or inseparable field the confidence
        is capped by what the best account actually earned, and the rivals go to
        :mod:`nyxara.njp.epistemic` — which is the organ that turns exactly this state into an
        experiment.
        """
        try:
            if self.competition is None:
                return
            from nyxara.njp.compete import Hypothesis, Verdict as CVerdict

            accounts: List[Any] = []
            if njp is None and thought.answer:
                # Her own answer as it stands, with the epistemic state that licenses it. Built
                # here so the competition runs on the ordinary path rather than only when a
                # cortex handed it a rival.
                from nyxara.njp.grounding import Answer
                njp = Answer(text=thought.answer, state=thought.epistemic,
                             confidence=thought.epistemic_confidence,
                             provenance=thought.provenance,
                             triples=list(getattr(getattr(thought, "derivation", None),
                                                  "support", None) or []))
            for seat, claim in (("cortex", cortex), ("njp", njp)):
                text = str(getattr(claim, "text", "") or "")
                if not text:
                    continue
                accounts.append(Hypothesis(
                    seat=seat, claim=text,
                    confidence=float(getattr(claim, "confidence", 0.0) or 0.0),
                    provenance=str(getattr(claim, "provenance", "") or "proposed"),
                    state=str(getattr(claim, "state", "") or "unknown"),
                    support=list(getattr(claim, "triples", None) or [])))

            seats = (self.fabric_proposes(thought.stimulus, k=1)
                     if self._gate("fabric_prior", True) else [])
            for name, probability in seats:
                accounts.append(Hypothesis(
                    seat="fabric", claim=name, confidence=float(probability),
                    # No provenance and no support, stated rather than implied: the head
                    # associates, it does not derive, and dressing that as a derivation would be
                    # the one thing this seat must never do.
                    provenance="proposed", state="unknown", support=[]))

            if len(accounts) < 2:
                return
            outcome = self.competition.compete(accounts)
            thought.competition = outcome
            if outcome.verdict in (CVerdict.UNSUPPORTED, CVerdict.TIED):
                best = outcome.scores[0].total if outcome.scores else 0.0
                thought.epistemic_confidence = min(thought.epistemic_confidence, float(best))
                if self.epistemic is not None and len(outcome.rivals) >= 2:
                    self.epistemic.compile(thought.stimulus, accounts)
        except Exception as exc:  # noqa: BLE001 — ranking is an organ, never a dependency
            log.debug("competition degraded: %s", exc)

    @staticmethod
    def _temper_by_novelty(base: float, thought: NJPThought,
                           *, damping: float = 0.75) -> float:
        """Discount a stated confidence by how unfamiliar the fabric found this turn.

        **This is the edge that made the fabric part of cognition.** Before it, the fabric grew on
        every turn and could not reach the answer path at all — ``"fabric" in
        getsource(_compose)`` was ``False``, and the only consumer of
        :attr:`~nyxara.njp.brain.NJPPercept.novelty` was :meth:`nyxara.njp.reasoner.NJPReasoner._temper`,
        a seat that a grounded turn never reaches. So a structural answer left here carrying the
        store's confidence and nothing else, and the manifold's opinion about whether she had ever
        seen anything like this was computed, recorded, and dropped.

        The rule is :meth:`~nyxara.njp.reasoner.NJPReasoner._temper`'s, deliberately unchanged —
        ``reported = base × (1 − novelty × damping)`` — so the two paths discount identically
        rather than growing two notions of what a novel turn is worth.

        **It can only ever lower.** ``novelty`` is in ``[0, 1]`` and ``damping`` is below 1, so the
        factor is never above 1 and a familiar turn is left exactly as it was. That direction is
        the whole safety property: the fabric is allowed to make her less sure of something she
        looked up, and is never allowed to make her more sure of anything. Growth earns its way
        into confidence by *removing* a discount as she comes to recognise the ground, which is
        also what makes it measurable — an unfamiliar turn and a familiar one now differ in a
        number that leaves this method.
        """
        try:
            percept = getattr(thought, "percept", None)
            if percept is None:
                return max(0.0, min(1.0, float(base)))
            factor = max(0.0, 1.0 - float(percept.novelty) * float(damping))
            return max(0.0, min(1.0, float(base) * factor))
        except Exception:  # noqa: BLE001 — an unreadable percept discounts nothing
            return max(0.0, min(1.0, float(base)))

    def _wonder_at_anomaly(self, thought: NJPThought) -> bool:
        """A confident answer on ground the substrate does not recognise becomes a question.

        The fabric's three existing edges all point the same way: they lower a confidence and stop.
        That is right as far as it goes and it is not learning — nothing follows from the fabric
        having been surprised, so a turn she answered firmly on a state she has never met leaves
        no trace that anything was odd about it.

        This is the other direction, and it is deliberately *not* the confidence: an anomaly
        produces a **question**, which cannot raise a confidence and cannot become an answer. It
        goes to :mod:`nyxara.njp.curiosity`, which already prices questions by value of
        information and already refuses to re-raise one that has been answered.

        Both halves are required. Low familiarity alone is most turns early on, and a confident
        answer alone is the ordinary case; it is the *combination* — sure of herself on ground she
        does not recognise — that is worth asking about.
        """
        if not self._gate("fabric_anomaly", True) or self.curiosity is None:
            return False
        try:
            percept = getattr(thought, "percept", None)
            anticipated = getattr(percept, "anticipated", None) if percept else None
            if anticipated is None or bool(getattr(anticipated, "trusted", False)):
                return False
            familiarity = float(getattr(anticipated, "familiarity", 1.0) or 0.0)
            if familiarity >= _ANOMALY_FAMILIARITY or not thought.answer:
                return False
            if float(getattr(thought, "epistemic_confidence", 0.0) or 0.0) < _ANOMALY_CONFIDENCE:
                return False

            from nyxara.njp.curiosity import Gap, Question
            subject = ""
            grounding = getattr(percept, "grounding", None)
            triples = list(getattr(grounding, "triples", None) or [])
            if triples:
                subject = str(getattr(triples[0], "subject", "") or "")
            raised = self.curiosity._raise(Question(
                text=(f"answered confidently on ground the substrate does not recognise: "
                      f"{thought.stimulus[:120]}"),
                gap=Gap.UNKNOWN, subject=subject,
                uncertainty=max(0.0, 1.0 - familiarity),
                stakes=0.4, cost=0.2, action="gather"))
            return raised is not None
        except Exception:  # noqa: BLE001 — an unaskable anomaly asks nothing
            return False

    def _temper_by_shape(self, base: float, thought: NJPThought) -> float:
        """Discount an answer by how the *form of reasoning* behind it has actually done.

        The consumer :mod:`nyxara.njp.genome` never had. It counts the shapes she reasons in,
        prices each by the MDL saving naming it would earn, and reports the ones that keep being
        wrong as :meth:`~nyxara.njp.genome.ReasoningGenome.liabilities` — and nothing anywhere
        read either list. So a recurring form with a measured record of failure went on producing
        answers at full confidence, and "a habit worth breaking" was a phrase in a docstring.

        This is a different question from :meth:`_temper_by_novelty` and from
        :class:`~nyxara.njp.core.Transitivity`, which is why it is a third number rather than a
        tweak to either. Novelty asks whether the substrate recognises the *situation*.
        Transitivity asks whether a *predicate* chains. This asks whether reasoning of this
        *shape* works — and the three come apart: inheriting through a stated ``is_a`` edge and
        chaining a predicate through itself may use the same predicate on equally familiar ground
        and fail at completely different rates.

        **Untested discounts nothing**, and that is the load-bearing half.
        :meth:`~nyxara.njp.genome.ReasoningGenome.reliability` returns ``None`` until a shape has
        been graded enough times to have a record, and a new inference must not be penalised for
        being new. Only measured failure lowers anything, and it can only lower — the evidence for
        an answer is the facts it was composed from, and this is evidence about her machinery.
        """
        try:
            if self.genome is None:
                return max(0.0, min(1.0, float(base)))
            measured = self.genome.reliability(getattr(thought, "trace", None))
            if measured is None:
                return max(0.0, min(1.0, float(base)))
            return max(0.0, min(1.0, float(base) * max(0.0, min(1.0, float(measured)))))
        except Exception:  # noqa: BLE001
            return max(0.0, min(1.0, float(base)))

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
                thought.epistemic_confidence = NJPBrain._temper_by_novelty(
                    float(answer.confidence), thought)
                # An answer built from proposals is a proposal. Carrying it here is what lets the
                # router tell "the record says X" from "something guessed X", which is the whole
                # difference between a contradiction and a disagreement.
                thought.provenance = getattr(answer, "provenance", thought.provenance)
                return
            # A live tie is its own state and must be read before the acknowledgement branch
            # below, which would otherwise label it KNOWN: that branch means "she is certain about
            # what she just recorded", and what she recorded here is a contradiction. Confidence
            # is left at zero deliberately — two answers she cannot separate is not partial
            # knowledge of either.
            if answer is not None and answer.conflicting:
                thought.epistemic = Epistemic.CONFLICTING
                thought.epistemic_confidence = 0.0
                thought.provenance = getattr(answer, "provenance", thought.provenance)
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
            # A derived answer prices its own chain: `core` multiplies each link's confidence by
            # that predicate's transitivity and caps the product below the KNOWN floor, so 0.448
            # for a two-hop inheritance is a real number about a real derivation. Starting from
            # `thought.confidence` instead threw it away — that field is the *gauntlet's* verdict,
            # correctly 0.0 whenever it abstains, which it does for every composed claim. So a
            # two-hop chain and a five-hop one both reported 0.0 and were indistinguishable, and
            # anything downstream that discounts a derived answer had nothing to discount.
            #
            # The prior is still never above what the derivation itself claimed, and
            # `revise_confidence` remains monotonic from there.
            prior = float(thought.confidence)
            derivation = getattr(thought, "derivation", None)
            if derivation is not None and getattr(derivation, "ok", False):
                prior = max(prior, float(getattr(derivation, "confidence", 0.0) or 0.0))
            confidence = revise_confidence(
                prior, independent_evidence=supports,
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
                if self.genome is not None:
                    self.genome.grade(getattr(thought, "trace", None),
                                      correct=float(correct) >= 0.5)
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
                            ("self_model", self.self_model), ("genome", self.genome),
                            ("meta", self.meta),
                            ("goals", self.goals), ("curiosity", self.curiosity),
                            ("attention", self.attention), ("readout", self.readout),
                            ("loop", self.loop),
                            ("concepts", self.genesis), ("universe", self.universe),
                            ("designer", self.designer), ("beliefs", self.beliefs),
                            ("rollout", self.rollout), ("falsify", self.killer),
                            ("propose", self.proposer),
                            ("metareason", self.metareason), ("predictive", self.predictive),
                            ("agency", self.agent), ("curriculum", self.curriculum),
                            ("calculate", self.calculator),
                            ("maths", self.mathematician),
                            ("mathsolver", self.solver),
                            ("field", self.field), ("learner", self.learner),
                            ("adversary", self.adversary),
                            ("cortex", self.cortex), ("router", self.router),
                            ("compiler", self.compiler),
                            ("blackbox", self.blackbox),
                            ("assumptions", self.assumptions),
                            ("index", self.index),
                            ("society", self.society),
                            ("evolution", self.evolution),
                            ("epistemic", self.epistemic), ("coding", self.coder),
                            ("language", self.language),
                            ("discourse", self.discourse)):
            if organ is None:
                continue                       # an organ that is off is ABSENT, not zeroed
            try:
                out[name] = organ.stats()
            except Exception:  # noqa: BLE001
                out[name] = {"error": "stats failed"}
        # Stage G scores `agency.success_rate`, and until `doing` existed the only thing under
        # that key was `Agent`, whose counters cannot move without an action vocabulary. Merged
        # rather than replaced: the planner over the dynamics model is still real and still
        # reported; what is added is the half that can actually be exercised.
        if self.doing is not None:
            try:
                got = self.doing.stats()
                out["doing"] = got
                block = out.get("agency")
                out["agency"] = {**block, **got} if isinstance(block, dict) else dict(got)
            except Exception:  # noqa: BLE001
                pass
        if self.memory is not None:
            try:
                # The whole surface, not just the count. `recalls_undecided` and
                # `recalls_rescored` are what separate "this arm never fires" from "this arm
                # fires and changes nothing", and those are different findings.
                out["memory"] = self.memory.stats()
            except Exception:  # noqa: BLE001
                pass
        return out

    def pipeline_report(self) -> Dict[str, Dict[str, str]]:
        """Every arrow of the cognitive loop: closed, open, or absent — and why.

        The point of this is the arrows that are **not** closed. Every other report here counts
        what happened; this one names what cannot. An organ that is gated off is *absent* rather
        than reported as a broken arrow, which is the same discipline :meth:`stats` keeps — the
        distinction between "off" and "failing" is one a report has no business blurring.

        Each entry is read from something real rather than declared: an arrow is closed when the
        organ that would carry it exists and the counter that proves it ran is non-zero, and open
        with a reason otherwise. A hardcoded "closed" would make this the sort of documentation
        the rest of the package is written against.
        """
        out: Dict[str, Dict[str, str]] = {}

        def arrow(name: str, organ: Any, reason_if_open: str = "",
                  ran: Optional[bool] = None) -> None:
            if organ is None:
                return                    # absent, not broken — the caller sees it is not listed
            if ran is False:
                out[name] = {"state": "open", "why": reason_if_open or "never ran"}
            else:
                out[name] = {"state": "closed", "why": ""}

        try:
            compiler = getattr(self.compiler, "stats", lambda: {})()
            grounded = getattr(self.grounder, "stats", lambda: {})()
            universe = getattr(self.universe, "stats", lambda: {})()
            predict = getattr(self.predictor, "stats", lambda: {})()

            arrow("reality→language", self.grounder)
            arrow("language→compiler", self.compiler,
                  "no turn has compiled into an executable operation",
                  ran=bool(compiler.get("executable", 0)))
            arrow("compiler→symbolic", self.world,
                  "no causal link has been recorded",
                  ran=bool(getattr(self.world, "stats", lambda: {})().get("causal_links", 0)))
            arrow("compiler→neural", self.fabric)
            arrow("symbolic⟷neural", self.epistemic,
                  "the two seats have not both spoken on one slot yet")
            arrow("→hypotheses", self.reasoner)
            arrow("qwythos→proposals", self.cortex,
                  "no cortex reachable — weights are not on disk",
                  ran=bool(getattr(self.cortex, "available", lambda: False)()))
            arrow("njp→proposals", self.metareason)
            arrow("→causal simulation", self.universe,
                  "no arrow is both fitted and oriented",
                  ran=bool(universe.get("oriented_relations", 0)))
            arrow("→counterfactual", self.universe,
                  "no intervention has been compiled and run",
                  ran=bool(compiler.get("executable", 0)))
            # Named where it actually lives. The encoder is `integrate.LearningLoop._encode_state`
            # — never `njp.predictive._encode_state`, which does not exist — and since Phase 7 it
            # is one of several the loop may feed, chosen by measurement rather than by the line
            # that was written first. The arrow is open while the promoted representation is still
            # the sixteen-bucket histogram, whose own docstring shows three unrelated words
            # sharing one state.
            arrow("→plan", self.predictive,
                  "the promoted state representation cannot carry identity — see "
                  "njp.evolution.ENCODERS and integrate.LearningLoop._encode_state",
                  ran=bool(getattr(getattr(self, "loop", None), "predictive_encoding", "")
                           not in ("", "buckets16", "buckets64")))
            arrow("test→prediction error", self.predictor,
                  "nothing has been scored", ran=bool(predict.get("scored", 0)))
            arrow("error→diagnosis", self.predictor,
                  "nothing has been diagnosed", ran=bool(predict.get("by_kind")))
            arrow("diagnosis→revision", self.predictor,
                  "no diagnosis has reached a repair", ran=bool(predict.get("repaired")))
            arrow("→concept", self.genesis)
            arrow("→program", self.noesis,
                  "the abstraction library has never been stepped",
                  ran=bool(getattr(self.noesis, "cycles", 0)))
            arrow("→meta-reasoning", self.metareason)
            arrow("meta→strategy learning", self.metareason,
                  "no kind has enough trials to name a winner",
                  ran=bool(getattr(self.metareason, "stats", lambda: {})().get("by_kind")))
            # Phase 6's two arrows. The first is the join nothing held: a grade that never
            # reaches the turn that chose is an outcome with no strategy attached, and no amount
            # of it makes a failure *mode*. The second is the half of the milestone that did not
            # exist — a weakness she can name and does not act on is a diagnosis, not a
            # curriculum.
            arrow("outcome→failure mode", self.blackbox,
                  "no graded answer has met a condition twice, so no join exists yet",
                  ran=bool(getattr(self.blackbox, "stats", lambda: {})().get("pairs", 0)))
            arrow("weakness→training", self.loop,
                  "nothing she can name as a weakness has become work she committed to",
                  ran=bool((getattr(self.loop, "totals", None) or {}).get(
                      "weaknesses_trained", 0)))
            # Phase 7. A knob is the field's business; this is the arrow that says whether
            # anything *structural* has ever survived a benchmark, and it is open until one has.
            arrow("society→verdict", self.society,
                  "the specialists have never been run over one claim in sequence",
                  ran=bool(getattr(self.society, "deliberations", 0)))
            arrow("progress→measured", self.index,
                  "the intelligence vector has never been read, so nothing knows whether she "
                  "is better than she was",
                  ran=bool(getattr(self.index, "measurements", 0)))
            arrow("goal→action", self.doing,
                  "nothing she can do has been chosen and run against a goal",
                  ran=bool(getattr(self.doing, "acted", 0)))
            arrow("weakness→rewire", self.evolution,
                  "no structural change has measured better than what she was written with",
                  ran=bool(getattr(self.evolution, "cognitive_rewires", 0)))
            # V.26. Open until she has actually been *talked to*: the organ holds no convention
            # until one is demonstrated and no referent until one is introduced, so a brain that
            # has only ever been handed isolated sentences reports this arrow open, correctly.
            arrow("turn→uptake", self.discourse,
                  "no turn has been read as part of a conversation",
                  ran=bool(getattr(self.discourse, "turn", 0)))
            arrow("↺", self.loop)
        except Exception:  # noqa: BLE001 — an unreadable organ is left out, not guessed at
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
        d: Dict[str, Any] = {"turns": self.turns, "fabric": self.fabric.to_dict(),
                             # Persisted with the fabric, and for the same reason `_cell_id` is a
                             # stable hash rather than a counter: a reloaded brain wired for cells
                             # it can no longer name is the one way persistence is worse than
                             # starting fresh. The lexicon IS the naming.
                             "lexicon": {str(k): v for k, v in self.lexicon.items()}}
        for name, organ in (("ledger", self.ledger), ("soulsync", self.soul),
                            ("grounding", self.grounder), ("world", self.world), ("predict", self.predictor), ("levels", self.levels),
                            ("discover", self.discoverer), ("reason", self.reasoner),
                            ("self_model", self.self_model), ("genome", self.genome),
                            ("meta", self.meta),
                            ("goals", self.goals), ("curiosity", self.curiosity),
                            ("attention", self.attention), ("readout", self.readout),
                            ("concepts", self.genesis), ("universe", self.universe),
                            ("designer", self.designer), ("beliefs", self.beliefs),
                            ("rollout", self.rollout), ("falsify", self.killer),
                            ("propose", self.proposer),
                            ("metareason", self.metareason), ("predictive", self.predictive),
                            ("agency", self.agent),
                            ("field", self.field), ("learner", self.learner),
                            ("adversary", self.adversary),
                            ("cortex", self.cortex), ("router", self.router),
                            ("compiler", self.compiler),
                            ("epistemic", self.epistemic), ("coding", self.coder),
                            ("language", self.language)):
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
            for cell, name in (d.get("lexicon") or {}).items():
                try:
                    self.lexicon.setdefault(int(cell), str(name))
                except (TypeError, ValueError):
                    continue
            if d.get("fabric"):
                self.fabric.load_dict(d["fabric"])
            if d.get("ledger") and self.ledger is not None:
                self.ledger.load_dict(d["ledger"])
            if d.get("soulsync") and self.soul is not None:
                self.soul.load_dict(d["soulsync"])
            if d.get("grounding") and self.grounder is not None:
                self.grounder.load_dict(d["grounding"])
            # The grammar she was taught. Re-derived from its demonstrations on the way in, so a
            # sidecar can make her forget a language and cannot make her believe a shape nobody
            # ever showed her.
            if d.get("language") and self.language is not None:
                self.language.load_dict(d["language"])
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
            if d.get("genome") and self.genome is not None:
                self.genome.load_dict(d["genome"])
            if d.get("meta") and self.meta is not None:
                self.meta.load_dict(d["meta"])
            if d.get("goals") and self.goals is not None:
                self.goals.load_dict(d["goals"])
            if d.get("curiosity") and self.curiosity is not None:
                self.curiosity.load_dict(d["curiosity"])
            if d.get("rollout") and self.rollout is not None:
                self.rollout.load_dict(d["rollout"])
            if d.get("falsify") and self.killer is not None:
                self.killer.load_dict(d["falsify"])
            if d.get("propose") and self.proposer is not None:
                self.proposer.load_dict(d["propose"])
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
