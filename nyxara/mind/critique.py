"""NYXARA · mind/critique.py — devil's advocate, biases, mental models, calibration.

Before NYXARA acts on a conclusion, this engine tries to *talk it out of it*. It is
the deliberate, adversarial half of thinking — the upgrade behind
``proposal.CritiqueStage``:

* **Cognitive-bias detection** — scans a claim and its support for the usual traps:
  overconfidence, confirmation bias, base-rate neglect, sunk-cost, availability,
  anchoring, survivorship.
* **Mental-model probes** — runs the claim through inversion, second-order effects,
  opportunity cost, margin of safety, and falsifiability, flagging the ones it fails.
* **Epistemic calibration** — checks whether the *stated* confidence is justified by
  the *evidence* (optionally against a learned :class:`~nyxara.mind.uncertainty.Calibrator`),
  and recommends a corrected confidence.
* **Devil's advocate** — generates the strongest counter-arguments and the conditions
  under which the claim would be wrong.

The result is a :class:`Critique` with a verdict (OK / CAUTION / OBJECT). An adapter
turns it into a critic callable for the proposal pipeline, so a strong objection
makes the kernel *abstain* and defer to the owner.

Depends on :mod:`mind.proposal`; optionally :mod:`mind.uncertainty`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "FindingKind",
    "Finding",
    "CritiqueContext",
    "Verdict",
    "Critique",
    "Critic",
    "as_proposal_critic",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #
class FindingKind(str, Enum):
    BIAS = "bias"
    MENTAL_MODEL = "mental_model"
    CALIBRATION = "calibration"


@dataclass
class Finding:
    kind: FindingKind
    name: str
    description: str
    severity: float           # [0,1]
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "name": self.name,
                "description": self.description, "severity": round(self.severity, 3),
                "suggestion": self.suggestion}


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass
class CritiqueContext:
    """Everything the critic needs to interrogate a claim."""

    claim: str
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    rationale: str = ""
    risk: float = 0.0
    reversibility: float = 1.0
    alternatives_considered: int = 0
    sample_size: Optional[int] = None
    base_rate: Optional[float] = None
    probabilistic: bool = False
    features: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_proposal(cls, proposal: Any) -> "CritiqueContext":
        meta = getattr(proposal, "metadata", {}) or {}
        return cls(
            claim=proposal.text() if hasattr(proposal, "text") else str(proposal),
            confidence=getattr(proposal, "confidence", 0.5),
            risk=getattr(proposal, "risk", 0.0),
            reversibility=getattr(proposal, "reversibility", 1.0),
            rationale=getattr(proposal, "rationale", ""),
            evidence=list(meta.get("evidence", [])),
            assumptions=list(meta.get("assumptions", [])),
            alternatives_considered=int(meta.get("alternatives_considered", 0)),
            sample_size=meta.get("sample_size"),
            base_rate=meta.get("base_rate"),
            probabilistic=bool(meta.get("probabilistic", False)),
            features=dict(meta.get("features", {})),
        )


# --------------------------------------------------------------------------- #
# Verdict & critique
# --------------------------------------------------------------------------- #
class Verdict(str, Enum):
    OK = "ok"
    CAUTION = "caution"
    OBJECT = "object"


@dataclass
class Critique:
    findings: List[Finding]
    counterarguments: List[str]
    probes: List[str]
    verdict: Verdict
    recommended_confidence: float
    objection: Optional[str] = None

    @property
    def severity(self) -> float:
        return max((f.severity for f in self.findings), default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict.value,
                "recommended_confidence": round(self.recommended_confidence, 3),
                "severity": round(self.severity, 3),
                "objection": self.objection,
                "findings": [f.to_dict() for f in self.findings],
                "counterarguments": self.counterarguments,
                "probes": self.probes}


# --------------------------------------------------------------------------- #
# The critic
# --------------------------------------------------------------------------- #
class Critic:
    """Runs bias/mental-model/calibration checks and plays devil's advocate."""

    _SUNK_COST = re.compile(
        r"already (invested|spent|put)|so far|come this far|can'?t (stop|quit|waste)|"
        r"waste(d)? (the |all )?(effort|money|time|work)", re.I)
    _AVAILABILITY = re.compile(
        r"\bi (just )?(saw|heard|read|remember)\b|the other day|recently i|in the news", re.I)
    _SECOND_ORDER_HINT = re.compile(
        r"second[- ]order|downstream|ripple|knock[- ]on|consequence|long[- ]term", re.I)

    def __init__(self, *, overconfidence_threshold: float = 0.8,
                 calibrator: Any = None) -> None:
        self.overconfidence_threshold = overconfidence_threshold
        self.calibrator = calibrator

    # ---- individual detectors ---- #
    def _overconfidence(self, c: CritiqueContext) -> Optional[Finding]:
        weak = len(c.evidence) < 2 or (c.sample_size is not None and c.sample_size < 5)
        if c.confidence >= self.overconfidence_threshold and weak:
            sev = _clamp(0.4 + (c.confidence - self.overconfidence_threshold) * 2)
            return Finding(FindingKind.CALIBRATION, "overconfidence",
                           f"confidence {c.confidence:.2f} rests on thin evidence "
                           f"({len(c.evidence)} item(s))", sev,
                           "lower confidence or gather more evidence")
        # learned-calibration cross-check
        if self.calibrator is not None and getattr(self.calibrator, "samples", 0) > 0:
            recal = self.calibrator.recalibrate(c.confidence)
            if c.confidence - recal > 0.2:
                return Finding(FindingKind.CALIBRATION, "miscalibration",
                               f"history says {c.confidence:.2f}-confidence claims are "
                               f"right only ~{recal:.0%} of the time", 0.6,
                               f"recalibrate toward {recal:.2f}")
        return None

    def _confirmation(self, c: CritiqueContext) -> Optional[Finding]:
        if c.alternatives_considered == 0 and c.evidence:
            return Finding(FindingKind.BIAS, "confirmation_bias",
                           "only confirming evidence cited; no alternatives weighed", 0.5,
                           "actively seek disconfirming evidence and a rival hypothesis")
        return None

    def _base_rate(self, c: CritiqueContext) -> Optional[Finding]:
        if c.probabilistic and c.base_rate is None:
            return Finding(FindingKind.BIAS, "base_rate_neglect",
                           "a probabilistic claim with no base rate considered", 0.5,
                           "anchor the estimate on the prior / base rate")
        if c.base_rate is not None and c.base_rate < 0.2 and c.confidence > 0.7:
            return Finding(FindingKind.BIAS, "base_rate_neglect",
                           f"high confidence ({c.confidence:.2f}) despite a low base rate "
                           f"({c.base_rate:.2f})", 0.6,
                           "let the low base rate pull the estimate down")
        return None

    def _sunk_cost(self, c: CritiqueContext) -> Optional[Finding]:
        if self._SUNK_COST.search(c.rationale):
            return Finding(FindingKind.BIAS, "sunk_cost",
                           "rationale appeals to past investment, not future value", 0.5,
                           "ignore sunk costs; decide on marginal future value")
        return None

    def _availability(self, c: CritiqueContext) -> Optional[Finding]:
        if c.features.get("anecdotal") or self._AVAILABILITY.search(c.rationale):
            return Finding(FindingKind.BIAS, "availability_bias",
                           "reasoning leans on a vivid, recent anecdote", 0.4,
                           "weigh frequency/statistics, not memorability")
        return None

    def _anchoring(self, c: CritiqueContext) -> Optional[Finding]:
        anchor = c.features.get("anchor")
        estimate = c.features.get("estimate")
        if anchor is not None and estimate is not None:
            try:
                if abs(estimate - anchor) <= 0.1 * (abs(anchor) + 1e-9):
                    return Finding(FindingKind.BIAS, "anchoring",
                                   "estimate clings to the initial anchor", 0.45,
                                   "re-derive the estimate ignoring the anchor")
            except TypeError:
                pass
        return None

    def _second_order(self, c: CritiqueContext) -> Optional[Finding]:
        if c.risk > 0.5 and c.reversibility < 0.4 \
                and not self._SECOND_ORDER_HINT.search(c.rationale):
            return Finding(FindingKind.MENTAL_MODEL, "second_order_effects",
                           "high-risk, hard-to-reverse, with no downstream analysis", 0.6,
                           "map second-order and long-term consequences")
        return None

    def _opportunity_cost(self, c: CritiqueContext) -> Optional[Finding]:
        if c.alternatives_considered == 0 and (c.risk > 0.3 or c.confidence >= 0.8):
            return Finding(FindingKind.MENTAL_MODEL, "opportunity_cost",
                           "no alternative was compared against", 0.4,
                           "what is the best thing NOT being done instead?")
        return None

    def _margin_of_safety(self, c: CritiqueContext) -> Optional[Finding]:
        if c.risk > 0.6 and c.reversibility < 0.3:
            return Finding(FindingKind.MENTAL_MODEL, "margin_of_safety",
                           "irreversible high-risk action with no safety buffer", 0.7,
                           "build in a margin of safety or a reversible first step")
        return None

    def _falsifiability(self, c: CritiqueContext) -> Optional[Finding]:
        if c.confidence > 0.7 and not c.evidence and not c.probabilistic:
            return Finding(FindingKind.MENTAL_MODEL, "unfalsifiable_confidence",
                           "confident assertion with no evidence and no stated test", 0.5,
                           "state what observation would prove this wrong")
        return None

    def _detectors(self) -> List[Callable[[CritiqueContext], Optional[Finding]]]:
        return [self._overconfidence, self._confirmation, self._base_rate,
                self._sunk_cost, self._availability, self._anchoring,
                self._second_order, self._opportunity_cost, self._margin_of_safety,
                self._falsifiability]

    # ---- devil's advocate ---- #
    def devils_advocate(self, c: CritiqueContext) -> List[str]:
        out = [
            f"Steelman the opposite: what is the strongest case that «{c.claim}» is false?",
            "Assume this turns out wrong — what is the single most likely reason?",
        ]
        if len(c.evidence) < 2:
            out.append(f"The claim rests on {len(c.evidence)} piece(s) of evidence; "
                       "one solid counterexample could overturn it.")
        if c.alternatives_considered == 0:
            out.append("State the strongest alternative explanation and why it loses.")
        if c.risk > 0.5:
            out.append("If this fails, who/what is harmed, and is that acceptable to the owner?")
        return out

    def probes(self, c: CritiqueContext) -> List[str]:
        return [
            "Inversion: what would guarantee this fails? Are we avoiding those?",
            "Second-order: and then what happens? (effects of the effects)",
            "Opportunity cost: what are we giving up by choosing this?",
            "Falsifiability: what observation would change our mind?",
        ]

    # ---- main ---- #
    def critique(self, c: CritiqueContext) -> Critique:
        findings = [f for f in (d(c) for d in self._detectors()) if f is not None]
        counter = self.devils_advocate(c)
        probes = self.probes(c)

        max_sev = max((f.severity for f in findings), default=0.0)
        if max_sev >= 0.7:
            verdict = Verdict.OBJECT
        elif max_sev >= 0.45 or len(findings) >= 3:
            verdict = Verdict.CAUTION
        else:
            verdict = Verdict.OK

        # recommended confidence: penalise by calibration/bias severity
        penalty = min(0.7, sum(f.severity for f in findings
                               if f.kind in (FindingKind.CALIBRATION, FindingKind.BIAS)) * 0.2)
        recommended = _clamp(c.confidence * (1.0 - penalty))

        objection = None
        if verdict is not Verdict.OK and findings:
            strongest = max(findings, key=lambda f: f.severity)
            objection = f"{strongest.name}: {strongest.description}"

        return Critique(findings=findings, counterarguments=counter, probes=probes,
                        verdict=verdict, recommended_confidence=recommended,
                        objection=objection)


# --------------------------------------------------------------------------- #
# Proposal-pipeline adapter
# --------------------------------------------------------------------------- #
def as_proposal_critic(critic: Optional[Critic] = None, *,
                       object_only: bool = True
                       ) -> Callable[[Any, Any], Optional[str]]:
    """Adapt a :class:`Critic` into a ``proposal.CritiqueStage`` critic callable.

    Returns ``None`` (no objection) unless the critique objects — so the proposal
    pipeline abstains only on a genuinely strong concern.
    """
    crit = critic or Critic()

    def _critic(proposal: Any, _ctx: Any) -> Optional[str]:
        c = CritiqueContext.from_proposal(proposal)
        result = crit.critique(c)
        if result.verdict is Verdict.OBJECT:
            return result.objection
        if not object_only and result.verdict is Verdict.CAUTION:
            return result.objection
        return None

    return _critic


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA critique self-test")
    print("=" * 70)

    critic = Critic()

    # 1) overconfident, thinly-evidenced claim -> calibration finding, lowered confidence
    c = CritiqueContext(claim="this stock will definitely double", confidence=0.95,
                        evidence=["a tip from a friend"], probabilistic=True)
    res = critic.critique(c)
    print(f"\noverconfident       : verdict={res.verdict.value} "
          f"rec_conf={res.recommended_confidence:.2f}")
    names = {f.name for f in res.findings}
    print(f"  findings          : {names}")
    assert "overconfidence" in names
    assert res.recommended_confidence < c.confidence

    # 2) irreversible high-risk action -> mental-model objections
    c2 = CritiqueContext(claim="delete the production database", confidence=0.9,
                         risk=0.9, reversibility=0.05, rationale="it's probably fine")
    res2 = critic.critique(c2)
    print(f"\nirreversible risk   : verdict={res2.verdict.value} "
          f"objection={res2.objection}")
    assert res2.verdict is Verdict.OBJECT
    assert any(f.name == "margin_of_safety" for f in res2.findings)
    assert any(f.name == "second_order_effects" for f in res2.findings)

    # 3) sunk-cost reasoning detected
    c3 = CritiqueContext(claim="keep funding the project", confidence=0.6,
                         rationale="we've already invested two years, can't waste it now")
    res3 = critic.critique(c3)
    print(f"\nsunk cost           : {[f.name for f in res3.findings]}")
    assert any(f.name == "sunk_cost" for f in res3.findings)

    # 4) base-rate neglect on a probabilistic claim with no base rate
    c4 = CritiqueContext(claim="the email is spam", confidence=0.85, probabilistic=True,
                         evidence=["has a link", "urgent tone"])
    res4 = critic.critique(c4)
    assert any(f.name == "base_rate_neglect" for f in res4.findings)
    print("base-rate neglect   : detected ✓")

    # 5) a careful, well-evidenced, reversible claim -> OK, no objection
    c5 = CritiqueContext(claim="ship the small bugfix", confidence=0.8,
                         evidence=["passing tests", "code review", "staged rollout"],
                         alternatives_considered=2, risk=0.1, reversibility=0.95,
                         rationale="reversible; considered alternatives; downstream effects mapped")
    res5 = critic.critique(c5)
    print(f"\ncareful claim       : verdict={res5.verdict.value} "
          f"objection={res5.objection}")
    assert res5.verdict is Verdict.OK and res5.objection is None

    # devil's advocate always argues the other side
    print("\ndevil's advocate    :")
    for cp in res2.counterarguments[:3]:
        print(f"  - {cp}")
    assert res2.counterarguments

    # 6) adapter into the proposal pipeline
    from nyxara.mind.proposal import (CritiqueStage, Proposal, ProposalContext,
                                      ProposalKind, StageVerdict)
    risky = Proposal(ProposalKind.ACTION, "wipe all backups", confidence=0.9,
                     risk=0.95, reversibility=0.02, rationale="probably fine")
    stage = CritiqueStage(critic=as_proposal_critic(critic))
    verdict = stage.evaluate(risky, ProposalContext())
    print(f"\npipeline critic     : {verdict.verdict.value} -> {verdict.reason}")
    assert verdict.verdict is StageVerdict.ABSTAIN

    print("\nALL SELF-TESTS PASSED ✓")
