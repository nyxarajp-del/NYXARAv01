"""NYXARA · njp/beliefs.py — Self-model 2.0: every belief carries its own case (🪞, NJP V.04).

:mod:`nyxara.njp.selfmodel` answers *"how good am I at X?"* — a number per capability, learned
from outcomes. That is a real self-model and it is the wrong grain for the question the Master
actually asked. "I can reason" is not self-knowledge. This module holds the other grain: **one
record per thing she believes**, and that record has to answer

* What do I know? — :meth:`BeliefLedger.known`
* What don't I know? — :meth:`BeliefLedger.unknown`
* Why do I believe this? — :meth:`BeliefLedger.why`
* What evidence supports it? — :attr:`Belief.evidence`, each piece with its own kind and weight
* What could falsify it? — :attr:`Belief.falsifier`, and a belief without one is *flagged*
* How reliable am I in this domain? — :meth:`BeliefLedger.reliability`, a real Brier score over
  her own past confidences
* Which reasoning process produced this? — :attr:`Belief.produced_by`

**The design decision that makes this more than a dictionary with extra fields**: a belief is
never *set*, only ever *revised*, and every revision appends to :attr:`Belief.history` with the
reason. So "why do you believe this?" is answered from a record rather than reconstructed
afterwards — and the reconstruction-afterwards version is precisely how a system ends up
confabulating justifications for beliefs it acquired some other way.

**Contradiction is tracked between beliefs, not resolved by fiat.** When two beliefs contradict,
both are marked and both lose confidence in proportion to the *other's* support, so the
better-evidenced one survives without either being silently deleted. What she used to think is
part of what she knows about herself.

**Calibration is the honesty check.** :meth:`BeliefLedger.reliability` compares the confidence she
*stated* against how those beliefs actually turned out, per domain, as a Brier score and a
signed bias. A domain where she is systematically overconfident is a fact about her that she can
read off her own ledger — and :meth:`BeliefLedger.temper` applies it, so a stated 0.9 in a domain
where 0.9 has meant 0.6 comes out as 0.6.

**Unsupported confidence is auditable.** :meth:`BeliefLedger.audit` returns the beliefs held above
a threshold with no hard evidence and no falsifier. That list is the honest answer to "what am I
asserting that I have not earned", and it is the input :mod:`nyxara.njp.field` uses to aim
curiosity at her own foundations rather than only at the outside world.

No LLM. Pure standard library.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "EvidenceKind",
    "Support",
    "Revision",
    "Belief",
    "Reliability",
    "BeliefLedger",
]


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _key(text: Any) -> str:
    return _norm(text).lower()[:200]


class EvidenceKind:
    """What *kind* of reason this is. Weight follows kind, and the ordering is the whole point."""

    PROOF = "proof"                 # derived, and the derivation checks
    OBSERVATION = "observation"     # she saw it happen
    PREDICTION = "prediction"       # she predicted it and the prediction held on held-out data
    TESTIMONY = "testimony"         # the Master said so
    INFERENCE = "inference"         # it follows from other beliefs
    CONSENSUS = "consensus"         # several sources agree — the weakest, and deliberately so

    # A hard reason can establish a belief on its own; a soft one can only ever corroborate.
    # Consensus is soft *because* it is popular: agreement between sources that share an origin
    # is one source counted many times, which is exactly how a confident error propagates.
    HARD = frozenset({PROOF, OBSERVATION, PREDICTION})
    WEIGHTS: Dict[str, float] = {
        PROOF: 1.0, OBSERVATION: 0.85, PREDICTION: 0.8,
        TESTIMONY: 0.6, INFERENCE: 0.45, CONSENSUS: 0.25,
    }


@dataclass
class Support:
    """One reason, with its kind, its weight and where it came from."""

    kind: str = EvidenceKind.INFERENCE
    detail: str = ""
    source: str = ""
    at: float = field(default_factory=time.time)

    @property
    def hard(self) -> bool:
        return self.kind in EvidenceKind.HARD

    @property
    def weight(self) -> float:
        return EvidenceKind.WEIGHTS.get(self.kind, 0.3)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail[:160], "source": self.source,
                "hard": self.hard, "weight": round(self.weight, 3)}


@dataclass
class Revision:
    """A moment at which what she believed changed, and why it changed."""

    at: float = field(default_factory=time.time)
    confidence: float = 0.0
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"at": round(self.at, 3), "confidence": round(self.confidence, 4),
                "why": self.why[:120]}


@dataclass
class Belief:
    """One thing she holds true, with the entire case for and against it attached."""

    claim: str = ""
    confidence: float = 0.5
    domain: str = "general"
    produced_by: str = ""                     # which reasoning process reached it
    evidence: List[Support] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)
    falsifier: str = ""                       # what observation would kill it
    explanation: str = ""                     # the causal story, when there is one
    history: List[Revision] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    outcome: Optional[bool] = None            # set once reality settled it, for calibration
    #: What the adversary made of this claim, and how often.
    #:
    #: The leak these close is real and was reproduced. `adversary.attack` returns an
    #: `AttackReport` and the caller drops it, so nothing durable records that a claim was tested
    #: and came back UNDECIDED. A claim in that state can then be re-held with TESTIMONY support
    #: by `field._record_beliefs` on any later turn where the Master restates it, and `earned`
    #: rises — so UNDECIDED becomes true by repetition, through a route that never consults the
    #: attack record. Being told a thing ten times is exactly what the SOFT_CEILING exists to
    #: refuse, and this is the hole in the floor beneath it.
    tested: int = 0
    undecided: int = 0
    last_stance: str = ""
    #: What `earned` had reached when the attack came back undecided. Testimony may not carry it
    #: past this until something other than testimony arrives.
    earned_at_test: Optional[float] = None
    #: How much record existed when the verdict landed. A retry is worth running once that number
    #: has grown and not before: the record is what settles an attack, so re-attacking against an
    #: unchanged record is the same four coin-flips at the same cost.
    observations_at_test: int = 0

    @property
    def key(self) -> str:
        return _key(self.claim)

    @property
    def hard_support(self) -> int:
        return sum(1 for e in self.evidence if e.hard)

    #: What soft evidence alone may ever earn. Testimony, inference and consensus can make a
    #: claim *credible*; nothing but a proof, an observation or a surviving prediction can make
    #: it established. Without this ceiling the diminishing-returns sum still converges high
    #: enough that ten tellings of a claim reach 0.96 — which is precisely the "repetition makes
    #: it true" failure the whole ledger exists to prevent, arrived at by arithmetic instead of
    #: by credulity. Mirrors the hard-source requirement in :mod:`nyxara.njp.truth`.
    SOFT_CEILING = 0.75

    @property
    def earned(self) -> float:
        """Confidence the evidence actually justifies, independent of what she *feels*.

        Diminishing returns per additional reason, and a ceiling that only hard evidence lifts:
        ten pieces of testimony are worth more than one and are still not a proof. The gap
        between this and :attr:`confidence` is what :meth:`BeliefLedger.audit` looks for.
        """
        if not self.evidence:
            return 0.0
        total = 0.0
        for i, support in enumerate(sorted(self.evidence, key=lambda e: -e.weight)):
            total += support.weight * (0.6 ** i)
        earned = total / (1.0 + total) * 1.6
        ceiling = 0.97 if self.hard_support else self.SOFT_CEILING
        earned = min(ceiling, earned)

        # A claim that was tested and came back undecided does not climb on being repeated.
        #
        # The SOFT_CEILING already refuses "ten tellings make it true" in general. This is the
        # narrower and sharper case: the adversary went looking, the record could not settle it,
        # and every subsequent telling is the same testimony arriving again. Hard evidence — a
        # proof, an observation, a surviving prediction — lifts it as it always could, because
        # that is a different kind of thing arriving rather than the same kind arriving twice.
        if self.undecided > 0 and not self.hard_support and self.earned_at_test is not None:
            earned = min(earned, float(self.earned_at_test))
        return earned

    @property
    def unsupported(self) -> bool:
        """Believed harder than the evidence warrants."""
        return self.confidence > self.earned + 0.2

    @property
    def falsifiable(self) -> bool:
        return bool(self.falsifier)

    @property
    def contested(self) -> bool:
        return bool(self.contradicts)

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim[:200], "confidence": round(self.confidence, 4),
                "earned": round(self.earned, 4), "domain": self.domain,
                "produced_by": self.produced_by, "hard_support": self.hard_support,
                "evidence": [e.to_dict() for e in self.evidence[:8]],
                "contradicts": self.contradicts[:6], "falsifier": self.falsifier[:160],
                "explanation": self.explanation[:200],
                "unsupported": self.unsupported, "falsifiable": self.falsifiable,
                "revisions": len(self.history), "outcome": self.outcome,
                "history": [r.to_dict() for r in self.history[-5:]]}


@dataclass
class Reliability:
    """How well her stated confidence in a domain has actually tracked the truth."""

    domain: str = ""
    samples: int = 0
    brier: Optional[float] = None      # mean squared error of confidence vs outcome; lower better
    bias: Optional[float] = None       # positive ⇒ overconfident
    accuracy: Optional[float] = None

    @property
    def calibrated(self) -> bool:
        return self.samples >= 5 and self.bias is not None and abs(self.bias) <= 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.domain, "samples": self.samples,
                "brier": None if self.brier is None else round(self.brier, 4),
                "bias": None if self.bias is None else round(self.bias, 4),
                "accuracy": None if self.accuracy is None else round(self.accuracy, 4),
                "calibrated": self.calibrated}


class BeliefLedger:
    """Every belief, its case, its contradictions, and how reliable she has been about its kind."""

    def __init__(self, *, capacity: int = 4000, self_model: Any = None) -> None:
        self.capacity = max(32, int(capacity))
        self.self_model = self_model            # optional njp.selfmodel.SelfModel
        self.beliefs: Dict[str, Belief] = {}
        self.open_questions: Dict[str, str] = {}    # question -> why it is open
        self.revisions = 0
        self.contradictions_found = 0
        self.retracted = 0
        # domain -> list of (stated confidence, outcome) once reality settled it.
        self._outcomes: Dict[str, List[Tuple[float, bool]]] = {}

    # ---- asserting --------------------------------------------------------- #
    def hold(self, claim: str, *, confidence: float = 0.5, domain: str = "general",
             produced_by: str = "", falsifier: str = "", explanation: str = "",
             why: str = "asserted") -> Optional[Belief]:
        """Hold a belief, or revise one already held. Never silently overwrites.

        A second assertion of something she already believes is *evidence about the claim*, not a
        replacement for the record — so the history keeps both moments and the confidence moves
        rather than being clobbered.
        """
        claim = _norm(claim)
        if not claim:
            return None
        confidence = min(1.0, max(0.0, float(confidence)))
        key = _key(claim)
        belief = self.beliefs.get(key)
        if belief is None:
            belief = Belief(claim=claim, confidence=confidence, domain=_norm(domain) or "general",
                            produced_by=_norm(produced_by), falsifier=_norm(falsifier),
                            explanation=_norm(explanation))
            belief.history.append(Revision(confidence=confidence, why=why))
            self.beliefs[key] = belief
            self._evict()
            return belief
        return self.revise(claim, confidence=confidence, why=why,
                           falsifier=falsifier, explanation=explanation,
                           produced_by=produced_by)

    def revise(self, claim: str, *, confidence: Optional[float] = None, why: str = "",
               falsifier: str = "", explanation: str = "",
               produced_by: str = "") -> Optional[Belief]:
        """Change a belief and record the change. The only route by which confidence moves."""
        belief = self.beliefs.get(_key(claim))
        if belief is None:
            return None
        if falsifier:
            belief.falsifier = _norm(falsifier)
        if explanation:
            belief.explanation = _norm(explanation)
        if produced_by:
            belief.produced_by = _norm(produced_by)
        if confidence is not None:
            new = min(1.0, max(0.0, float(confidence)))
            if abs(new - belief.confidence) > 1e-9:
                belief.confidence = new
                belief.history.append(Revision(confidence=new, why=_norm(why) or "revised"))
                belief.history = belief.history[-32:]
                self.revisions += 1
        belief.updated = time.time()
        return belief

    def support(self, claim: str, kind: str, detail: str = "",
                source: str = "") -> Optional[Belief]:
        """Attach a reason, and let the confidence follow the evidence rather than the other way.

        Confidence after a new reason is the *maximum* of what she felt and what the evidence now
        earns — never a free increment. A belief cannot be talked up by repeating it, which is
        the failure mode this whole ledger exists to make visible.
        """
        belief = self.beliefs.get(_key(claim))
        if belief is None:
            return None
        belief.evidence.append(Support(kind=_norm(kind) or EvidenceKind.INFERENCE,
                                       detail=_norm(detail), source=_norm(source)))
        belief.evidence = belief.evidence[-16:]
        earned = belief.earned
        if earned > belief.confidence:
            self.revise(claim, confidence=earned, why=f"evidence: {kind}")
        else:
            belief.updated = time.time()
        return belief

    def contradict(self, claim: str, other: str, *, why: str = "") -> Tuple[Optional[Belief],
                                                                           Optional[Belief]]:
        """Mark two beliefs as incompatible and let the better-evidenced one win.

        Neither is deleted. Each loses confidence in proportion to the *other's* earned support,
        so a well-evidenced claim barely moves against a bare assertion, and two equally bare
        assertions both collapse — which is the correct outcome, because between them she has no
        reason to prefer either.
        """
        a = self.beliefs.get(_key(claim))
        b = self.beliefs.get(_key(other))
        if a is None or b is None or a.key == b.key:
            return a, b
        if b.key not in a.contradicts:
            a.contradicts.append(b.key)
        if a.key not in b.contradicts:
            b.contradicts.append(a.key)
        self.contradictions_found += 1
        reason = _norm(why) or "contradicted"
        a_earned, b_earned = a.earned, b.earned
        self.revise(claim, confidence=a.confidence * (1.0 - min(0.9, b_earned)),
                    why=f"{reason}: {b.claim[:40]}")
        self.revise(other, confidence=b.confidence * (1.0 - min(0.9, a_earned)),
                    why=f"{reason}: {a.claim[:40]}")
        return a, b

    def record_attack(self, claim: str, report: Any, *,
                      observations: int = 0) -> Optional[Belief]:
        """Store what the adversary made of a claim, so the verdict survives the turn.

        `AttackReport` was returned and dropped. The consequence is not that she forgot an opinion
        — it is that the same claim is re-attacked from scratch on every encounter, each attempt
        an independent coin-flip against a thin record, and that an UNDECIDED verdict has no way
        of constraining what happens to the belief afterwards.

        Only the undecided case pins ``earned_at_test``. SURVIVED needs no ceiling and REFUTED is
        handled by retraction, which is a different and stronger thing than a cap.
        """
        try:
            belief = self.beliefs.get(_key(claim))
            if belief is None or report is None:
                return None
            # Derived from what the attacks actually found, not read off a field. `AttackReport`
            # sets `stance` only when the attacks were skipped; on a real run the verdict lives in
            # the attacks themselves, and reading the empty field recorded every outcome as "no
            # stance" — which meant the undecided record never populated and the whole mechanism
            # below it was inert.
            stance = str(getattr(report, "stance", "") or "")
            if not stance:
                from nyxara.njp.adversary import Stance as _S
                if getattr(report, "refuted_by", None):
                    stance = _S.REFUTED
                elif getattr(report, "survived", 0) and not getattr(report, "undecided", 0):
                    stance = _S.SURVIVED
                else:
                    stance = _S.UNDECIDED
            belief.tested += 1
            belief.last_stance = stance
            from nyxara.njp.adversary import Stance
            if stance == Stance.UNDECIDED:
                belief.undecided += 1
                belief.observations_at_test = max(int(observations),
                                                  belief.observations_at_test)
                if belief.earned_at_test is None:
                    belief.earned_at_test = float(belief.earned)
            belief.history.append(Revision(
                confidence=belief.confidence,
                why=f"attacked: {stance or 'no verdict'}"))
            return belief
        except Exception:  # noqa: BLE001 — an unrecordable attack records nothing
            return None

    def retract(self, claim: str, *, why: str = "falsified") -> Optional[Belief]:
        """Drive a belief to zero and keep it. What she used to think is data about her."""
        belief = self.revise(claim, confidence=0.0, why=_norm(why))
        if belief is not None:
            belief.outcome = False
            self._record_outcome(belief, False)
            self.retracted += 1
        return belief

    def settle(self, claim: str, *, true: bool, why: str = "observed") -> Optional[Belief]:
        """Reality decided. Records the outcome for calibration — the point of the whole ledger."""
        belief = self.beliefs.get(_key(claim))
        if belief is None:
            return None
        self._record_outcome(belief, bool(true))
        belief.outcome = bool(true)
        return self.revise(claim, confidence=1.0 if true else 0.0, why=_norm(why))

    def _record_outcome(self, belief: Belief, outcome: bool) -> None:
        # The confidence recorded is the one she held *before* reality answered — grading her
        # against the confidence she adopted afterwards would score her at perfect calibration
        # forever, which is the classic way this measurement gets quietly broken.
        stated = belief.history[-1].confidence if belief.history else belief.confidence
        for revision in reversed(belief.history):
            if revision.why not in ("observed", "falsified"):
                stated = revision.confidence
                break
        self._outcomes.setdefault(belief.domain, []).append((stated, outcome))
        self._outcomes[belief.domain] = self._outcomes[belief.domain][-500:]
        if self.self_model is not None:
            try:
                self.self_model.observe(belief.domain, 1.0 if outcome else 0.0)
            except Exception:  # noqa: BLE001 — the self-model is optional
                pass

    # ---- the seven questions ------------------------------------------------ #
    def why(self, claim: str) -> Dict[str, Any]:
        """The full case: what she believes, how strongly, on what, against what, and since when."""
        belief = self.beliefs.get(_key(claim))
        if belief is None:
            return {"claim": _norm(claim), "known": False,
                    "reason": "no such belief is held"}
        rel = self.reliability(belief.domain)
        return {
            "claim": belief.claim, "known": True,
            "confidence": round(belief.confidence, 4),
            "earned": round(belief.earned, 4),
            "tempered": round(self.temper(belief.confidence, belief.domain), 4),
            "produced_by": belief.produced_by or "unrecorded",
            "evidence": [e.to_dict() for e in belief.evidence],
            "hard_support": belief.hard_support,
            "explanation": belief.explanation,
            "falsifier": belief.falsifier or "",
            "falsifiable": belief.falsifiable,
            "contradicts": [self.beliefs[k].claim for k in belief.contradicts
                            if k in self.beliefs][:6],
            "history": [r.to_dict() for r in belief.history[-8:]],
            "domain_reliability": rel.to_dict(),
            "unsupported": belief.unsupported,
        }

    def known(self, *, threshold: float = 0.6) -> List[Belief]:
        """What she actually knows — held confidently *and* with the support to justify it."""
        return sorted((b for b in self.beliefs.values()
                       if b.confidence >= threshold and not b.unsupported),
                      key=lambda b: b.confidence, reverse=True)

    def unknown(self) -> List[Dict[str, Any]]:
        """The honest inventory of ignorance: open questions, contested and unsupported beliefs.

        Three genuinely different kinds of not-knowing, kept apart because they call for
        different repairs — ask, adjudicate, or go and find evidence.
        """
        out: List[Dict[str, Any]] = []
        for question, why in self.open_questions.items():
            out.append({"kind": "open_question", "subject": question, "detail": why})
        for belief in self.beliefs.values():
            if belief.contested:
                out.append({"kind": "contested", "subject": belief.claim[:120],
                            "detail": f"conflicts with {len(belief.contradicts)} other belief(s)"})
            elif belief.unsupported and belief.confidence >= 0.5:
                out.append({"kind": "unsupported", "subject": belief.claim[:120],
                            "detail": f"held at {belief.confidence:.2f}, "
                                      f"evidence earns {belief.earned:.2f}"})
        return out

    def ask(self, question: str, why: str = "") -> str:
        """Register something she knows she does not know."""
        question = _norm(question)
        if question:
            self.open_questions[question] = _norm(why) or "unanswered"
        return question

    def answered(self, question: str) -> bool:
        return self.open_questions.pop(_norm(question), None) is not None

    def reliability(self, domain: str = "general") -> Reliability:
        """Brier score and signed bias over her own past confidences in this domain.

        Brier is the mean squared error between stated confidence and outcome — 0 is perfect,
        0.25 is what you get by always saying "maybe". Bias is mean confidence minus mean
        outcome: positive means she talks a better game than she plays.
        """
        domain = _norm(domain) or "general"
        rows = self._outcomes.get(domain, [])
        out = Reliability(domain=domain, samples=len(rows))
        if not rows:
            return out
        out.brier = sum((c - (1.0 if o else 0.0)) ** 2 for c, o in rows) / len(rows)
        out.bias = (sum(c for c, _ in rows) / len(rows)
                    - sum(1.0 for _, o in rows if o) / len(rows))
        out.accuracy = sum(1.0 for c, o in rows if (c >= 0.5) == o) / len(rows)
        return out

    def temper(self, confidence: float, domain: str = "general") -> float:
        """Correct a stated confidence by how this domain has actually gone for her.

        Applied only once there is enough history to mean anything — with three samples the
        "correction" is noise, and a mind that lets noise adjust its confidence has made itself
        less calibrated, not more.
        """
        rel = self.reliability(domain)
        if rel.samples < 5 or rel.bias is None:
            return float(confidence)
        return min(1.0, max(0.0, float(confidence) - rel.bias))

    def audit(self, *, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Beliefs asserted above ``threshold`` that have not earned it. Her own weak foundations."""
        out: List[Dict[str, Any]] = []
        for belief in self.beliefs.values():
            if belief.confidence < threshold:
                continue
            problems: List[str] = []
            if belief.unsupported:
                problems.append("confidence exceeds evidence")
            if not belief.hard_support:
                problems.append("no hard evidence")
            if not belief.falsifiable:
                problems.append("no falsifier")
            if problems:
                out.append({"claim": belief.claim[:140], "domain": belief.domain,
                            "confidence": round(belief.confidence, 4),
                            "earned": round(belief.earned, 4), "problems": problems})
        out.sort(key=lambda r: (-len(r["problems"]), -r["confidence"]))
        return out

    # ---- housekeeping -------------------------------------------------------- #
    def _evict(self) -> None:
        if len(self.beliefs) <= self.capacity:
            return
        # Least useful first: low confidence, no evidence, untouched longest. Never evicts a
        # belief that is contested — an unresolved conflict is exactly what must not be forgotten.
        ranked = sorted(self.beliefs.values(),
                        key=lambda b: (b.contested, b.confidence + 0.2 * len(b.evidence),
                                       b.updated))
        for belief in ranked[: len(self.beliefs) - self.capacity]:
            if belief.contested:
                continue
            self.beliefs.pop(belief.key, None)

    def stats(self) -> Dict[str, Any]:
        beliefs = list(self.beliefs.values())
        audited = self.audit()
        return {
            "beliefs": len(beliefs),
            "known": len(self.known()),
            "open_questions": len(self.open_questions),
            "contested": sum(1 for b in beliefs if b.contested),
            "unsupported": sum(1 for b in beliefs if b.unsupported),
            "falsifiable": sum(1 for b in beliefs if b.falsifiable),
            "with_hard_evidence": sum(1 for b in beliefs if b.hard_support),
            "revisions": self.revisions,
            "contradictions_found": self.contradictions_found,
            "retracted": self.retracted,
            "settled": sum(1 for b in beliefs if b.outcome is not None),
            "audit_flags": len(audited),
            "domains": {d: self.reliability(d).to_dict() for d in list(self._outcomes)[:8]},
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beliefs": [{
                "claim": b.claim, "confidence": b.confidence, "domain": b.domain,
                "produced_by": b.produced_by, "falsifier": b.falsifier,
                "explanation": b.explanation, "contradicts": b.contradicts,
                "outcome": b.outcome, "created": b.created, "updated": b.updated,
                "evidence": [{"kind": e.kind, "detail": e.detail, "source": e.source, "at": e.at}
                             for e in b.evidence],
                "history": [{"at": r.at, "confidence": r.confidence, "why": r.why}
                            for r in b.history[-16:]],
            } for b in self.beliefs.values()],
            "open_questions": dict(self.open_questions),
            "outcomes": {d: [[c, bool(o)] for c, o in rows]
                         for d, rows in self._outcomes.items()},
            "revisions": self.revisions,
            "contradictions_found": self.contradictions_found,
            "retracted": self.retracted,
        }

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.beliefs = {}
            for row in (d.get("beliefs") or []):
                claim = _norm(row.get("claim"))
                if not claim:
                    continue
                belief = Belief(
                    claim=claim, confidence=float(row.get("confidence", 0.5)),
                    domain=_norm(row.get("domain")) or "general",
                    produced_by=_norm(row.get("produced_by")),
                    falsifier=_norm(row.get("falsifier")),
                    explanation=_norm(row.get("explanation")),
                    contradicts=[str(c) for c in (row.get("contradicts") or [])],
                    created=float(row.get("created", time.time())),
                    updated=float(row.get("updated", time.time())))
                outcome = row.get("outcome")
                belief.outcome = None if outcome is None else bool(outcome)
                for ev in (row.get("evidence") or []):
                    belief.evidence.append(Support(
                        kind=str(ev.get("kind", EvidenceKind.INFERENCE)),
                        detail=str(ev.get("detail", "")), source=str(ev.get("source", "")),
                        at=float(ev.get("at", 0.0))))
                for rv in (row.get("history") or []):
                    belief.history.append(Revision(
                        at=float(rv.get("at", 0.0)),
                        confidence=float(rv.get("confidence", 0.0)),
                        why=str(rv.get("why", ""))))
                self.beliefs[belief.key] = belief
            self.open_questions = {str(k): str(v)
                                   for k, v in (d.get("open_questions") or {}).items()}
            self._outcomes = {str(dom): [(float(c), bool(o)) for c, o in rows]
                              for dom, rows in (d.get("outcomes") or {}).items()}
            self.revisions = int(d.get("revisions", 0))
            self.contradictions_found = int(d.get("contradictions_found", 0))
            self.retracted = int(d.get("retracted", 0))
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves her with no beliefs, not false ones
            pass
