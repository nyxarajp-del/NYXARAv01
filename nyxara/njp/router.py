"""NYXARA · njp/router.py — who answers, and what a disagreement is worth (🚦, NJP V.06).

Two seats can now answer a turn: her cortex (:mod:`nyxara.njp.cortex`) and NJP itself. This module
decides between them, and it is deliberately much smaller than that sentence suggests — because
most of the work was already done and doing it again would have made her worse.

**It does not classify problems.** :class:`~nyxara.njp.metareason.ProblemClassifier` already scores a
problem on independent signals — arithmetic surface, causal connectives, self-reference,
contradiction with something believed, plain unknown-ness — and reports the winner *with its
margin*, so a narrow margin says "genuinely mixed" rather than being rounded away. And
:class:`~nyxara.njp.metareason.MetaReasoner` already picks the reasoning *strategy* with a UCB1
bandit over real outcomes. A router that re-derived either from keywords would be a second, worse
opinion competing with a learned one. So :meth:`Router.classify` delegates, and this module never
looks at the words itself.

**It does not re-open the hole :mod:`nyxara.njp.relevance` closed.** A greeting must not reach
physics. :class:`~nyxara.njp.relevance.CognitivePolicy` decides which pathways a *speech act*
permits, and the cortex is a REASON/SIMULATE faculty like any other — so "How are you NYXARA?" does
not get a 9B model generating causal hypotheses about it, for exactly the reason it does not get a
pendulum law.

**Which seat goes first is measured, not typed in.** The order comes from
:class:`~nyxara.njp.selfmodel.MetaLearner` — one bandit arm per problem kind, ``cortex`` and ``njp``
as its strategies — and from :meth:`~nyxara.njp.selfmodel.SelfModel.trust`, which discounts a stated
confidence by how that faculty has actually performed. "When do I trust the cortex" is therefore a
number that moves with outcomes. A hand-written preference table would have been a guess frozen at
the moment it was written, and it would have been wrong in a way nothing could ever notice.

**A disagreement is information, not a tie to break.** This is the part worth the module. When the
cortex says *A* and NJP says *B*, picking a winner throws away the only genuinely new thing that
happened: two systems with different access to the world reached different conclusions, so
something distinguishes them, and it is findable. So:

* NJP's answer contradicts the **record** — a grounded, non-proposed fact — and the cortex simply
  loses. That is not a disagreement between peers; it is a hypothesis meeting evidence.
* Otherwise the clash is *classified* — different assumptions, different evidence, or a different
  causal model — and becomes a question for :mod:`nyxara.njp.epistemic` to design an experiment
  around. No winner is declared, and confidence goes **down**, because two accounts that cannot
  both be right are grounds for less certainty, never more.

Every entry point is fail-soft: with no classifier, no policy or no self-model attached, the router
degrades to a documented default and says which one it used, rather than pretending to a decision it
did not make.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nyxara.njp.grounding import Epistemic, Provenance

log = logging.getLogger("nyxara.njp.router")

__all__ = [
    "Seat",
    "DisagreementKind",
    "Verdict",
    "Routing",
    "Arbitration",
    "Router",
]


class Seat:
    """Who can answer. Three, and they are not interchangeable.

    :attr:`FABRIC` is the one that does not speak, and the distinction is not a limitation being
    apologised for — it is what the seat actually is. The fabric produces a
    :class:`~nyxara.njp.manifold.Prediction`: which cells it expects to fire, with a margin and a
    familiarity. There is no path by which that becomes the string ``"water"``. A seat that
    emitted text on its behalf would be inventing the content and attributing it to the substrate,
    which is the precise failure :mod:`nyxara.njp.voice` refuses for the language surface.

    So the fabric asserts the one thing it is actually entitled to assert: **whether it recognises
    this situation at all.** That is not an answer and it is not nothing — a confident reply on
    ground the substrate has never met is exactly the confident-but-wrong case the honesty gates
    exist to catch, and until this seat existed nothing in arbitration could see it.

    Giving the fabric a genuine third *hypothesis* needs a readout that emits symbols rather than
    cell ids — :class:`~nyxara.njp.learn.Readout` is where that would come from — and it is
    deliberately not attempted here.
    """

    CORTEX = "cortex"      # proposes; never sourced, never statable on its own word
    NJP = "njp"            # grounds, recalls, derives, predicts — and owns the record
    FABRIC = "fabric"      # anticipates; asserts recognition, never content

    ALL = (CORTEX, NJP, FABRIC)

    #: The seats that can produce a text answer, and therefore the only ones routing orders or
    #: arbitration can hand a turn to. Read by :meth:`Router.route`, which used to read ``ALL`` —
    #: the split exists so adding a non-speaking seat could not silently put the fabric in a
    #: running order for a question it has no way to answer.
    SPEAKING = (CORTEX, NJP)


class Verdict:
    """What arbitration concluded. ``DISAGREEMENT`` is an outcome, not a failure to decide."""

    AGREE = "agree"                  # both answered, and to the same thing
    CONTRADICTION = "contradiction"  # the cortex clashed with the RECORD; the record wins
    DISAGREEMENT = "disagreement"    # both plausible, neither grounded — this is work, not a tie
    CORTEX_ONLY = "cortex_only"      # only the cortex spoke; stays a proposal
    NJP_ONLY = "njp_only"            # only NJP spoke
    SILENT = "silent"                # neither had anything


class DisagreementKind:
    """*Why* two seats disagree — the classification the experiment is designed against."""

    EVIDENCE = "different_evidence"        # they are not looking at the same facts
    CAUSAL_MODEL = "different_causal_model"  # same facts, different arrows between them
    ASSUMPTIONS = "different_assumptions"  # one holds a condition the other does not
    #: A speaking seat is confident and the fabric does not recognise the situation. Not a clash
    #: about *what is true* — the fabric never claimed anything about that — but about whether
    #: there is any ground under the answer at all.
    RECOGNITION = "unrecognised_situation"
    UNCLASSIFIED = "unclassified"          # it differs and the record cannot say why

    ALL = (EVIDENCE, CAUSAL_MODEL, ASSUMPTIONS, RECOGNITION, UNCLASSIFIED)


#: Problem kind → the NJP faculty whose posterior speaks for the NJP seat. ``reasoning`` is the
#: honest fallback: :meth:`SelfModel._measured` walks the hierarchy, so a kind with no better match
#: is discounted by the parent's record rather than by an invented capability that has never been
#: observed and therefore never discounts anything.
_NJP_CAPABILITY: Dict[str, str] = {
    "factual": "grounding",
    "causal": "reasoning",
    "symbolic": "reasoning",
    "empirical": "prediction",
    "contradiction": "grounding",
    "introspective": "recall",
}

#: The cortex is tracked as ONE faculty. It is a single system that is good or bad at proposing,
#: and splitting it per problem kind would spread thin evidence across arms that each then have too
#: few observations to discount anything at all.
_CORTEX_CAPABILITY = "reasoning:cortex"

#: How much of a confidence gap is needed before the stronger side is preferred outright. Below it
#: the two are treated as equally supported, which is the honest reading and the one that produces
#: a question instead of a coin flip.
_DECISIVE_MARGIN = 0.15

#: A clash costs confidence. Two accounts that cannot both be right are grounds for less certainty
#: in whichever survives, and this is the floor of that reduction.
_CLASH_PENALTY = 0.2

#: Below this an answer is already hedged, and the fabric's dissent would be punishing an honest
#: hedge rather than catching an overconfident claim. Set at the point where a reply starts being
#: read as an assertion rather than an offer.
_CONFIDENT_FLOOR = 0.5


def _norm(text: Any) -> str:
    """Loose text identity — enough to tell "the same answer" from "a different answer"."""
    return " ".join(str(text or "").strip().lower().split())


def _claim_text(claim: Any) -> str:
    """The answer text out of whatever shape the caller passed.

    Every read here is guarded because the two things being reconciled are an ``Answer`` this
    package built and *whatever the cortex handed back* — and a property that raises on access
    would otherwise take the turn down from inside the one method whose job is to reconcile
    disagreement.
    """
    for attribute in ("text", "claim", "answer"):
        try:
            got = getattr(claim, attribute, None)
        except Exception:  # noqa: BLE001 — an attribute that raises is an attribute that is absent
            continue
        if got:
            return str(got)
    return str(claim or "") if isinstance(claim, str) else ""


def _confidence(claim: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(getattr(claim, "confidence", default))))
    except Exception:  # noqa: BLE001
        return default


def _provenance(claim: Any) -> str:
    try:
        return str(getattr(claim, "provenance", Provenance.OBSERVED) or Provenance.OBSERVED)
    except Exception:  # noqa: BLE001 — an unreadable provenance is the weakest one, never the best
        return Provenance.PROPOSED


def _is_record(claim: Any) -> bool:
    """Is this the *record* speaking — something grounded rather than proposed?

    Both halves are required. ``KNOWN`` alone is not enough now that a proposal can be held
    confidently, and ``provenance`` alone is not enough because an observed-but-weak claim is still
    only a belief. A hypothesis loses to the record; it does not lose to another hypothesis.
    """
    try:
        state = str(getattr(claim, "state", ""))
    except Exception:  # noqa: BLE001
        return False
    return state == Epistemic.KNOWN and _provenance(claim) != Provenance.PROPOSED


@dataclass
class Routing:
    """Which seats may answer this turn, in which order, and on what basis."""

    kind: str = "factual"
    mixed: bool = False
    margin: float = 0.0
    seats: Tuple[str, ...] = (Seat.NJP,)
    cortex_permitted: bool = False
    #: May the cortex be used to extract structure (relations, laws) from this turn's text? A
    #: different question from whether it may hypothesise — see :meth:`Router.extraction_permitted`.
    extraction_permitted: bool = False
    allowed_pathways: Tuple[str, ...] = ()
    #: Why this order — "measured" when a posterior decided it, "default" when nothing was attached.
    basis: str = "default"
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "mixed": self.mixed, "margin": round(self.margin, 4),
                "seats": list(self.seats), "cortex_permitted": self.cortex_permitted,
                "extraction_permitted": self.extraction_permitted,
                "allowed_pathways": list(self.allowed_pathways),
                "basis": self.basis, "why": self.why}


@dataclass
class Arbitration:
    """What two answers came to, and — when they came to nothing — what would settle it."""

    verdict: str = Verdict.SILENT
    winner: str = ""                     # Seat.* , or empty when nothing won
    text: str = ""
    confidence: float = 0.0
    provenance: str = Provenance.PROPOSED
    disagreement: str = ""               # DisagreementKind.*, only on a DISAGREEMENT
    #: What to go and find out. Populated on DISAGREEMENT, and read by :mod:`nyxara.njp.epistemic`.
    question: str = ""
    #: The rival accounts, kept whole so an experiment can be designed against them.
    rivals: List[str] = field(default_factory=list)
    why: str = ""
    at: float = field(default_factory=time.time)

    @property
    def settled(self) -> bool:
        """Did anything win? A disagreement is a real outcome and is deliberately not settled."""
        return bool(self.winner)

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "winner": self.winner, "text": self.text,
                "confidence": round(self.confidence, 4), "provenance": self.provenance,
                "disagreement": self.disagreement or None, "question": self.question or None,
                "rivals": list(self.rivals), "settled": self.settled, "why": self.why}


class Router:
    """Routes a turn between her cortex and NJP, and turns their clashes into questions."""

    def __init__(self, *, meta: Any = None, policy: Any = None, reader: Any = None,
                 self_model: Any = None, learner: Any = None) -> None:
        self._meta = meta
        self._meta_tried = meta is not None
        self._policy = policy
        self._policy_tried = policy is not None
        self._reader = reader
        self._reader_tried = reader is not None
        self.self_model = self_model
        self.learner = learner
        self.routed = 0
        self.arbitrated = 0
        self.verdicts: Dict[str, int] = {}
        self.disagreements: Dict[str, int] = {}
        self.questions: List[str] = []
        #: Turns where the fabric had a trusted prediction behind a confident answer, and turns
        #: where it had none. The ratio is what says whether the substrate is actually coming to
        #: recognise the ground she works on, or merely growing.
        self.fabric_corroborations = 0
        self.fabric_dissents = 0

    # ---- collaborators, all optional --------------------------------------- #
    def _get_meta(self) -> Any:
        if not self._meta_tried:
            self._meta_tried = True
            try:
                from nyxara.njp.metareason import MetaReasoner
                self._meta = MetaReasoner()
            except Exception:  # noqa: BLE001
                self._meta = None
        return self._meta

    def _get_policy(self) -> Any:
        if not self._policy_tried:
            self._policy_tried = True
            try:
                from nyxara.njp.relevance import CognitivePolicy
                self._policy = CognitivePolicy()
            except Exception:  # noqa: BLE001
                self._policy = None
        return self._policy

    def _get_reader(self) -> Any:
        if not self._reader_tried:
            self._reader_tried = True
            try:
                from nyxara.njp.relevance import SpeechActReader
                self._reader = SpeechActReader()
            except Exception:  # noqa: BLE001
                self._reader = None
        return self._reader

    # ---- classification: delegated, never re-derived ------------------------ #
    def classify(self, problem: str, *, context: Optional[Dict[str, Any]] = None) -> Any:
        """What kind of problem this is — answered by :mod:`nyxara.njp.metareason`, not here.

        This method exists so callers have one door, and it is a *delegation*: the scoring rule,
        the margin and the learned strategy selection all live in the classifier that already had
        them. Re-deriving any of it here would put a keyword list in competition with a bandit that
        learns from outcomes, and the keyword list would win the arguments it should lose.
        """
        meta = self._get_meta()
        classifier = getattr(meta, "classifier", None) if meta is not None else None
        for candidate in (classifier, meta):
            if candidate is not None and hasattr(candidate, "classify"):
                try:
                    return candidate.classify(problem, context=context)
                except TypeError:
                    try:
                        return candidate.classify(problem)
                    except Exception:  # noqa: BLE001
                        break
                except Exception:  # noqa: BLE001
                    break
        try:
            from nyxara.njp.metareason import Classification
            return Classification()
        except Exception:  # noqa: BLE001
            return None

    # ---- may the cortex speak at all? -------------------------------------- #
    def cortex_permitted(self, text: str, *, act: Any = None) -> Tuple[bool, Tuple[str, ...], str]:
        """``(permitted, allowed_pathways, why)`` for this turn's speech act.

        The cortex is a REASON/SIMULATE faculty. Granting it a pathway the policy withholds from
        every other faculty would make it the one way world knowledge could still reach a greeting —
        the precise failure ``relevance.py`` exists to prevent, rebuilt behind a bigger model.
        """
        policy = self._get_policy()
        if policy is None:
            return True, (), "no cognitive policy attached — the cortex is not gated this turn"
        if act is None:
            reader = self._get_reader()
            if reader is not None:
                try:
                    act = reader.read(text)
                except Exception:  # noqa: BLE001
                    act = None
        try:
            from nyxara.njp.relevance import Pathway
            allowed = tuple(policy.allowed(act))
            permitted = bool({Pathway.REASON, Pathway.SIMULATE, Pathway.EXPERIMENT} & set(allowed))
            why = ("this act permits reasoning" if permitted
                   else f"this act permits only {', '.join(allowed) or 'nothing'} — "
                        "the cortex is a reasoning faculty and is not consulted")
            return permitted, allowed, why
        except Exception as exc:  # noqa: BLE001
            return True, (), f"policy check failed ({exc}); the cortex is not gated this turn"

    def extraction_permitted(self, text: str, *, act: Any = None) -> Tuple[bool, str]:
        """May the cortex be used to EXTRACT structure from this turn's text?

        A separate permission from :meth:`cortex_permitted`, and separate for a reason. Pulling
        relations and causal claims out of a sentence the Master just said is a *grounding*
        activity, not a reasoning one — so it is gated on ``Pathway.GROUNDING``, which
        ``CognitivePolicy`` grants to statements and withholds from greetings exactly as it does
        for every other faculty. Folding the two permissions into one would mean either that the
        World Model Generator could never run (statements permit no reasoning) or that hypothesis
        generation leaked into turns that permit no reasoning. Both are worse than two names.
        """
        policy = self._get_policy()
        if policy is None:
            return True, "no cognitive policy attached — extraction is not gated this turn"
        if act is None:
            reader = self._get_reader()
            if reader is not None:
                try:
                    act = reader.read(text)
                except Exception:  # noqa: BLE001
                    act = None
        try:
            from nyxara.njp.relevance import Pathway
            allowed = tuple(policy.allowed(act))
            permitted = Pathway.GROUNDING in allowed
            return permitted, ("this act permits grounding" if permitted
                               else f"this act permits only {', '.join(allowed) or 'nothing'} — "
                                    "nothing factual may be extracted from it")
        except Exception as exc:  # noqa: BLE001
            return True, f"policy check failed ({exc}); extraction is not gated this turn"

    # ---- seat order: measured ---------------------------------------------- #
    def _seat_score(self, seat: str, kind: str) -> Optional[float]:
        """How well this seat has actually done on this kind of problem, or None if untested."""
        if self.self_model is None:
            return None
        capability = (_CORTEX_CAPABILITY if seat == Seat.CORTEX
                      else _NJP_CAPABILITY.get(kind, "reasoning"))
        try:
            # trust() discounts a stated 1.0 by the measured level, so this IS the posterior mean
            # for that faculty — and it returns the input unchanged when there is not yet enough
            # evidence to discount, which is exactly the "untested is not bad" rule.
            return float(self.self_model.trust(capability, 1.0))
        except Exception:  # noqa: BLE001
            return None

    def route(self, text: str, *, act: Any = None,
              context: Optional[Dict[str, Any]] = None) -> Routing:
        """Which seats answer this turn, strongest-first."""
        self.routed += 1
        classification = self.classify(text, context=context)
        kind = str(getattr(classification, "kind", "factual") or "factual")
        routing = Routing(kind=kind,
                          mixed=bool(getattr(classification, "mixed", False)),
                          margin=float(getattr(classification, "margin", 0.0) or 0.0))

        permitted, allowed, why = self.cortex_permitted(text, act=act)
        routing.cortex_permitted = permitted
        routing.extraction_permitted = self.extraction_permitted(text, act=act)[0]
        routing.allowed_pathways = allowed
        routing.why = why
        if not permitted:
            routing.seats = (Seat.NJP,)
            return routing

        # The bandit gets first say: it is the thing that learns from outcomes.
        chosen = None
        if self.learner is not None:
            arm = f"seat:{kind}"
            try:
                for seat in Seat.SPEAKING:
                    self.learner.register(arm, seat, seat)
                strategy = self.learner.choose(arm)
                chosen = getattr(strategy, "name", None)
            except Exception:  # noqa: BLE001
                chosen = None
        if chosen in Seat.SPEAKING:
            routing.seats = (chosen, *(s for s in Seat.SPEAKING if s != chosen))
            routing.basis = "bandit"
            routing.why = f"{why}; seat order chosen by the bandit for '{kind}'"
            return routing

        cortex_score = self._seat_score(Seat.CORTEX, kind)
        njp_score = self._seat_score(Seat.NJP, kind)
        if cortex_score is None or njp_score is None:
            # Nothing measured yet. NJP first is the honest default rather than a preference: it is
            # the seat that owns the record, and a proposal is only worth weighing against one.
            routing.seats = (Seat.NJP, Seat.CORTEX)
            routing.basis = "default"
            routing.why = f"{why}; nothing measured yet, so the seat that owns the record goes first"
            return routing
        routing.seats = ((Seat.CORTEX, Seat.NJP) if cortex_score > njp_score
                         else (Seat.NJP, Seat.CORTEX))
        routing.basis = "measured"
        routing.why = (f"{why}; measured on '{kind}': cortex {cortex_score:.2f}, "
                       f"njp {njp_score:.2f}")
        return routing

    # ---- arbitration: a clash is information -------------------------------- #
    def arbitrate(self, question: str, *, cortex: Any = None, njp: Any = None,
                  fabric: Any = None) -> Arbitration:
        """Reconcile the speaking seats — or refuse to, and say what would settle it.

        ``fabric`` is the substrate's :class:`~nyxara.njp.manifold.Prediction` for this turn, and
        it is optional in the signature for a reason that is not politeness: every caller that
        existed before the seat did keeps its exact behaviour, so the seat can be measured against
        a baseline rather than replacing one. Passing ``None`` is the old function.

        It does not compete for the answer — see :class:`Seat`. It reviews the one the speaking
        seats produced, and can only ever *lower* what comes out.
        """
        self.arbitrated += 1
        cortex_text, njp_text = _claim_text(cortex), _claim_text(njp)
        out = Arbitration(rivals=[t for t in (cortex_text, njp_text) if t])

        if not cortex_text and not njp_text:
            out.verdict, out.why = Verdict.SILENT, "neither seat had anything to say"
            return self._record(self._review_by_fabric(out, fabric))
        if not cortex_text:
            out.verdict, out.winner = Verdict.NJP_ONLY, Seat.NJP
            out.text, out.confidence = njp_text, _confidence(njp)
            out.provenance, out.why = _provenance(njp), "only NJP answered"
            return self._record(self._review_by_fabric(out, fabric))
        if not njp_text:
            out.verdict, out.winner = Verdict.CORTEX_ONLY, Seat.CORTEX
            out.text, out.confidence = cortex_text, _confidence(cortex)
            # Unopposed is not the same as supported. A cortex answer with nothing to weigh it
            # against is still a proposal, and saying so here is what stops "nobody objected" from
            # becoming a source.
            out.provenance = Provenance.PROPOSED
            out.why = "only the cortex answered — unopposed, and still a proposal"
            return self._record(self._review_by_fabric(out, fabric))

        if _norm(cortex_text) == _norm(njp_text):
            out.verdict, out.winner = Verdict.AGREE, Seat.NJP
            out.text = njp_text
            out.confidence = max(_confidence(cortex), _confidence(njp))
            # Agreement does not launder provenance. If one side only proposed it, the pair is a
            # proposal that happens to be popular.
            out.provenance = Provenance.weakest(_provenance(cortex), _provenance(njp))
            out.why = "both seats reached the same answer"
            return self._record(self._review_by_fabric(out, fabric))

        # They differ. Is one of them the record?
        if _is_record(njp):
            out.verdict, out.winner = Verdict.CONTRADICTION, Seat.NJP
            out.text, out.confidence = njp_text, _confidence(njp)
            out.provenance = _provenance(njp)
            out.why = ("the cortex contradicted a grounded fact — that is a hypothesis meeting "
                       "evidence, not two peers disagreeing")
            return self._record(self._review_by_fabric(out, fabric))

        # Neither is grounded: this is the case worth keeping whole.
        out.verdict = Verdict.DISAGREEMENT
        out.disagreement = self._classify_disagreement(cortex, njp)
        out.provenance = Provenance.PROPOSED
        gap = abs(_confidence(cortex) - _confidence(njp))
        out.confidence = max(0.0, max(_confidence(cortex), _confidence(njp)) - _CLASH_PENALTY)
        out.question = self._question(question, cortex_text, njp_text, out.disagreement)
        out.why = (f"neither account is grounded and they differ ({out.disagreement}); "
                   f"confidence lowered by the clash"
                   + ("" if gap >= _DECISIVE_MARGIN
                      else "; the two are held about equally, so no side is preferred"))
        self.questions.append(out.question)
        return self._record(self._review_by_fabric(out, fabric))

    @staticmethod
    def _support_keys(claim: Any) -> set:
        """Which facts an account rests on, as comparable keys. Empty when it says."""
        keys = set()
        for attribute in ("triples", "support", "evidence"):
            for item in (getattr(claim, attribute, None) or ()):
                try:
                    if hasattr(item, "as_tuple"):
                        keys.add(tuple(str(p).lower() for p in item.as_tuple()))
                    else:
                        keys.add(_norm(item))
                except Exception:  # noqa: BLE001
                    continue
        return {k for k in keys if k}

    @classmethod
    def _classify_disagreement(cls, cortex: Any, njp: Any) -> str:
        """*Why* they differ, read off the two accounts rather than guessed.

        Ordered so the cheapest decisive check runs first. ``UNCLASSIFIED`` is returned honestly
        whenever nothing separates them — an experiment designed against a made-up reason would be
        worse than one designed against an admitted unknown.
        """
        cortex_support, njp_support = cls._support_keys(cortex), cls._support_keys(njp)
        if cortex_support and njp_support and not (cortex_support & njp_support):
            return DisagreementKind.EVIDENCE
        cortex_condition = _norm(getattr(cortex, "condition", "") or getattr(cortex, "under", ""))
        njp_condition = _norm(getattr(njp, "condition", "") or getattr(njp, "under", ""))
        if cortex_condition != njp_condition and (cortex_condition or njp_condition):
            return DisagreementKind.ASSUMPTIONS
        if cortex_support and njp_support and (cortex_support & njp_support):
            # Same facts, different conclusion: the arrows between them are what differs.
            return DisagreementKind.CAUSAL_MODEL
        return DisagreementKind.UNCLASSIFIED

    @staticmethod
    def _question(asked: str, cortex_text: str, njp_text: str, kind: str) -> str:
        """The question a clash becomes. Phrased as what to OBSERVE, never as who to believe."""
        head = f"{asked.strip()} — " if asked else ""
        tail = {
            DisagreementKind.EVIDENCE: "they are resting on different facts; which set holds?",
            DisagreementKind.ASSUMPTIONS: "they assume different conditions; which one obtains?",
            DisagreementKind.CAUSAL_MODEL: ("they read the same facts with different causes; "
                                            "what would occur under one and not the other?"),
            DisagreementKind.UNCLASSIFIED: "what observation would separate them?",
        }.get(kind, "what observation would separate them?")
        return f"{head}'{cortex_text}' or '{njp_text}': {tail}"

    def _review_by_fabric(self, out: Arbitration, fabric: Any) -> Arbitration:
        """Let the substrate dissent from a confident answer on ground it does not recognise.

        **What this deliberately does not do is discount by novelty.** That signal is already
        spent: :meth:`nyxara.njp.brain.NJPBrain._temper_by_novelty` multiplies a grounded answer's
        confidence by ``1 − novelty × damping`` on every turn. Applying familiarity again here
        would charge one turn twice for one unfamiliar substrate, and the second charge would look
        like independent corroboration of the first.

        So this reads a different property. :attr:`~nyxara.njp.manifold.Prediction.trusted` is the
        manifold's own statement about whether it had *enough transitions and enough separation*
        to make a forward prediction at all — an evidence question, not a similarity score, and
        one it answers ``False`` for honestly rather than by producing a weak guess. A speaking
        seat that is confident where the substrate could not form a prediction is asserting
        something about a situation nothing in her dynamics has any purchase on.

        Only above :data:`_CONFIDENT_FLOOR`, because a hedged answer on unfamiliar ground is
        already behaving correctly and lowering it further would punish exactly the honesty this
        module wants. And only ``AGREE`` / ``*_ONLY`` verdicts are reviewed: a ``DISAGREEMENT`` has
        already been penalised and already carries a question, and a ``CONTRADICTION`` was settled
        by the record — which outranks the substrate, since the record is evidence and recognition
        is not.
        """
        try:
            if fabric is None:
                return out
            if out.verdict not in (Verdict.AGREE, Verdict.NJP_ONLY, Verdict.CORTEX_ONLY):
                return out
            if out.confidence < _CONFIDENT_FLOOR:
                return out
            # Absent is not the same as untrusted. `getattr(..., False)` alone would read anything
            # that is not a Prediction — a stub, a mock, a None-shaped placeholder — as a fabric
            # saying "I do not recognise this", and manufacture a dissent out of a caller's
            # malformed argument. A seat that cannot be read abstains.
            if not hasattr(fabric, "trusted"):
                return out
            if bool(getattr(fabric, "trusted", False)):
                self.fabric_corroborations += 1
                return out
            self.fabric_dissents += 1
            out.disagreement = DisagreementKind.RECOGNITION
            out.confidence = max(0.0, out.confidence - _CLASH_PENALTY)
            reason = str(getattr(fabric, "reason", "") or "no trusted prediction")
            out.why += (f"; the fabric does not recognise this situation ({reason}), "
                        f"so the answer stands on structure alone and is held less firmly")
            # Deliberately still `settled`: the record answered and recognition does not unseat
            # it. What changed is how firmly it is held, which is the honest size of the doubt.
            return out
        except Exception:  # noqa: BLE001 — an unreadable prediction dissents from nothing
            return out

    def _record(self, out: Arbitration) -> Arbitration:
        self.verdicts[out.verdict] = self.verdicts.get(out.verdict, 0) + 1
        if out.disagreement:
            self.disagreements[out.disagreement] = self.disagreements.get(out.disagreement, 0) + 1
        return out

    # ---- learning from the outcome ------------------------------------------ #
    def observe(self, routing: Routing, seat: str, *, success: float) -> None:
        """Feed one turn's outcome back to the bandit and the self-model. Never raises.

        Without this the "measured" order would be a measurement nobody ever took. It is the whole
        reason the seat order can be called learned rather than merely computed.
        """
        success = max(0.0, min(1.0, float(success)))
        if self.learner is not None:
            try:
                self.learner.reward(f"seat:{routing.kind}", success)
            except Exception:  # noqa: BLE001
                pass
        if self.self_model is not None:
            capability = (_CORTEX_CAPABILITY if seat == Seat.CORTEX
                          else _NJP_CAPABILITY.get(routing.kind, "reasoning"))
            try:
                self.self_model.observe(capability, success)
            except Exception:  # noqa: BLE001
                pass

    def stats(self) -> Dict[str, Any]:
        return {"routed": self.routed, "arbitrated": self.arbitrated,
                "verdicts": dict(self.verdicts), "disagreements": dict(self.disagreements),
                "open_questions": len(self.questions),
                "fabric_corroborations": self.fabric_corroborations,
                "fabric_dissents": self.fabric_dissents}
