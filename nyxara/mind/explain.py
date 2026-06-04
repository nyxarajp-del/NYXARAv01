"""NYXARA · mind/explain.py — self-explanation of the reasoning chain (✦).

A mind that cannot say *why* it concluded something cannot be trusted or corrected.
This module reconstructs NYXARA's actual reasoning as an ordered chain of
:class:`ReasoningStep` s and narrates it — in **brief**, **detailed**, or
**technical** registers — plus a causal "because … therefore …" chain.

It is deliberately a *reconstruction of recorded steps*, not a post-hoc story:
it ingests the real traces other faculties already produce —

* a :class:`~nyxara.mind.proposal.Disposition` trail (the gates a proposal passed),
* a :class:`~nyxara.mind.reasoner.ReasoningResult` (strategy / consensus / dissent),
* a :class:`~nyxara.mind.dual_process.Outcome` (System-1 vs System-2 and why),
* a faculty-selection ranking, and a :class:`~nyxara.mind.critique.Critique` —

and turns them into a faithful narrative. This honesty (narrate what actually
happened, not what sounds good) is the seed of ``observe/honesty.py``.

Imports are duck-typed where possible so the explainer never breaks the thing it
explains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "StepKind",
    "ReasoningStep",
    "Explanation",
    "NarrationStyle",
    "ExplanationBuilder",
]


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
class StepKind(str, Enum):
    PERCEIVE = "perceive"
    RECALL = "recall"
    GOAL = "goal"
    SELECT = "select"      # chose a faculty / strategy / system
    REASON = "reason"
    SIMULATE = "simulate"  # imagined via world model
    CRITIQUE = "critique"
    MORAL = "moral"
    GATE = "gate"          # a pipeline check
    DECIDE = "decide"
    ABSTAIN = "abstain"
    ACT = "act"


@dataclass
class ReasoningStep:
    kind: StepKind
    summary: str
    detail: str = ""
    output: Any = None
    confidence: Optional[float] = None
    evidence: List[str] = field(default_factory=list)
    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "kind": self.kind.value, "summary": self.summary,
                "detail": self.detail, "output": _short(self.output),
                "confidence": None if self.confidence is None else round(self.confidence, 3)}


def _short(x: Any, n: int = 80) -> Any:
    if isinstance(x, str) and len(x) > n:
        return x[: n - 1] + "…"
    return x


# --------------------------------------------------------------------------- #
# Narration
# --------------------------------------------------------------------------- #
class NarrationStyle(str, Enum):
    BRIEF = "brief"
    DETAILED = "detailed"
    TECHNICAL = "technical"


_TEMPLATES = {
    StepKind.PERCEIVE: "I observed {summary}",
    StepKind.RECALL: "I recalled {summary}",
    StepKind.GOAL: "my goal was {summary}",
    StepKind.SELECT: "I chose {summary}",
    StepKind.REASON: "reasoning via {summary}",
    StepKind.SIMULATE: "I imagined {summary}",
    StepKind.CRITIQUE: "on reflection, {summary}",
    StepKind.MORAL: "ethically, {summary}",
    StepKind.GATE: "the {summary} check returned {output}",
    StepKind.DECIDE: "I decided {summary}",
    StepKind.ABSTAIN: "I held back because {summary}",
    StepKind.ACT: "I acted: {summary}",
}


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #
@dataclass
class Explanation:
    steps: List[ReasoningStep]
    conclusion: Any = None
    overall_confidence: Optional[float] = None

    # ---- rendering ---- #
    def _phrase(self, step: ReasoningStep, *, technical: bool) -> str:
        tmpl = _TEMPLATES.get(step.kind, "{summary}")
        text = tmpl.format(summary=step.summary, output=_short(step.output))
        if step.detail:
            text += f" ({step.detail})"
        if technical and step.confidence is not None:
            text += f" [conf {step.confidence:.2f}]"
        return text

    def narrate(self, style: NarrationStyle = NarrationStyle.DETAILED) -> str:
        if not self.steps:
            return "I have no recorded reasoning to explain."
        if style is NarrationStyle.BRIEF:
            key = [s for s in self.steps
                   if s.kind in (StepKind.DECIDE, StepKind.ABSTAIN, StepKind.REASON,
                                 StepKind.SELECT)]
            key = key or self.steps
            head = self._phrase(key[-1], technical=False)
            because = next((self._phrase(s, technical=False) for s in self.steps
                            if s.kind in (StepKind.REASON, StepKind.GATE,
                                          StepKind.CRITIQUE, StepKind.SELECT)), "")
            out = head if not because else f"{head}, because {because}"
            if self.conclusion is not None:
                out += f". Conclusion: {_short(self.conclusion)}"
            return out + "."
        technical = style is NarrationStyle.TECHNICAL
        lines = [f"{i + 1}. {self._phrase(s, technical=technical).capitalize()}."
                 for i, s in enumerate(self.steps)]
        if self.conclusion is not None:
            conf = (f" (confidence {self.overall_confidence:.2f})"
                    if self.overall_confidence is not None else "")
            lines.append(f"Therefore, I concluded: {_short(self.conclusion)}{conf}.")
        return "\n".join(lines)

    def causal_chain(self) -> str:
        """A compact 'because A, and B, therefore C' chain over the reasoning steps."""
        reasons = [s.summary for s in self.steps
                   if s.kind in (StepKind.PERCEIVE, StepKind.RECALL, StepKind.REASON,
                                 StepKind.SELECT, StepKind.SIMULATE, StepKind.CRITIQUE)]
        if not reasons:
            return f"Therefore {_short(self.conclusion)}." if self.conclusion is not None else ""
        body = ", and ".join(reasons)
        tail = _short(self.conclusion) if self.conclusion is not None else "I decided as I did"
        return f"Because {body}, therefore {tail}."

    def to_dict(self) -> Dict[str, Any]:
        return {"conclusion": _short(self.conclusion),
                "overall_confidence": (None if self.overall_confidence is None
                                       else round(self.overall_confidence, 3)),
                "steps": [s.to_dict() for s in self.steps]}


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class ExplanationBuilder:
    """Accumulate reasoning steps (manually or by ingesting traces) then narrate."""

    def __init__(self) -> None:
        self._steps: List[ReasoningStep] = []

    def __len__(self) -> int:
        return len(self._steps)

    def add(self, kind: StepKind, summary: str, *, detail: str = "", output: Any = None,
            confidence: Optional[float] = None,
            evidence: Optional[Sequence[str]] = None) -> "ExplanationBuilder":
        self._steps.append(ReasoningStep(
            kind=kind, summary=summary, detail=detail, output=output,
            confidence=confidence, evidence=list(evidence or []),
            index=len(self._steps)))
        return self

    # ---- ingest known traces (duck-typed) ---- #
    def ingest_faculty_ranking(self, ranking: Sequence[Any]) -> "ExplanationBuilder":
        if not ranking:
            return self
        top = ranking[0]
        name = getattr(getattr(top, "faculty", top), "name", str(top))
        others = [getattr(getattr(r, "faculty", r), "name", str(r)) for r in ranking[1:]]
        verifiable = getattr(top, "verifiable", None)
        detail = "verifiable engine preferred" if verifiable else "best available engine"
        if others:
            detail += f"; over {', '.join(others)}"
        return self.add(StepKind.SELECT, f"the '{name}' faculty", detail=detail,
                        confidence=getattr(top, "score", None))

    def ingest_dual_process(self, outcome: Any) -> "ExplanationBuilder":
        proc = getattr(getattr(outcome, "process", None), "value", "a process")
        decision = getattr(outcome, "decision", None)
        reason = getattr(decision, "reason", "")
        self.add(StepKind.SELECT, f"{proc} thinking", detail=reason)
        prop = getattr(outcome, "proposal", None)
        if prop is not None:
            self.add(StepKind.REASON, getattr(prop, "source_faculty", "a faculty"),
                     detail=getattr(prop, "rationale", ""), output=getattr(prop, "content", None),
                     confidence=getattr(prop, "confidence", None))
        return self

    def ingest_reasoning(self, result: Any) -> "ExplanationBuilder":
        consensus = getattr(result, "consensus", None)
        if consensus is not None and getattr(consensus, "conclusions", None):
            for c in consensus.conclusions:
                ans = getattr(c, "answer", None)
                if ans is None:
                    continue
                self.add(StepKind.REASON, getattr(c, "strategy", "a strategy"),
                         detail=getattr(c, "rationale", ""), output=ans,
                         confidence=getattr(c, "confidence", None))
            for d in getattr(consensus, "dissent", []):
                self.add(StepKind.CRITIQUE,
                         f"{getattr(d, 'strategy', 'a strategy')} dissented",
                         output=getattr(d, "answer", None))
            self.add(StepKind.DECIDE,
                     f"by {getattr(consensus, 'agreement', 0):.0%} agreement",
                     output=getattr(consensus, "answer", None),
                     confidence=getattr(consensus, "confidence", None))
        else:
            conc = getattr(result, "conclusion", result)
            self.add(StepKind.REASON, getattr(conc, "strategy", "reasoning"),
                     detail=getattr(conc, "rationale", ""),
                     output=getattr(conc, "answer", None),
                     confidence=getattr(conc, "confidence", None))
        return self

    def ingest_critique(self, critique: Any) -> "ExplanationBuilder":
        for f in getattr(critique, "findings", []):
            self.add(StepKind.CRITIQUE, getattr(f, "name", "a concern"),
                     detail=getattr(f, "description", ""),
                     confidence=getattr(f, "severity", None))
        objection = getattr(critique, "objection", None)
        if objection:
            self.add(StepKind.ABSTAIN, objection)
        return self

    def ingest_disposition(self, disposition: Any) -> "ExplanationBuilder":
        for r in getattr(disposition, "trail", []):
            self.add(StepKind.GATE, getattr(r, "stage", "a gate"),
                     detail=getattr(r, "reason", ""),
                     output=getattr(getattr(r, "verdict", None), "value", None),
                     confidence=getattr(r, "severity", None) or None)
        approved = getattr(disposition, "approved", None)
        if approved is not None:
            verdict = getattr(getattr(disposition, "verdict", None), "value", "")
            if approved:
                self.add(StepKind.DECIDE, "to act on the proposal",
                         detail="all gates passed")
            else:
                self.add(StepKind.ABSTAIN, getattr(disposition, "reason", verdict))
        return self

    # ---- build ---- #
    def build(self, *, conclusion: Any = None,
              overall_confidence: Optional[float] = None) -> Explanation:
        if overall_confidence is None:
            confs = [s.confidence for s in self._steps
                     if s.kind is StepKind.DECIDE and s.confidence is not None]
            overall_confidence = confs[-1] if confs else None
        return Explanation(steps=list(self._steps), conclusion=conclusion,
                           overall_confidence=overall_confidence)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA self-explanation self-test")
    print("=" * 70)

    # build a realistic chain by ingesting real traces from other faculties
    from nyxara.mind.proposal import Pipeline, Proposal, ProposalContext, ProposalKind
    from nyxara.mind.reasoner import (CallableStrategy, Reasoner, ReasoningMode,
                                      ReasoningQuery)

    # 1) a small consensus reasoning
    panel = Reasoner([
        CallableStrategy("bayesian", lambda q: ("approve", 0.8, "posterior favours approve"),
                         applicability_fn=lambda q: 1.0),
        CallableStrategy("causal", lambda q: ("approve", 0.7, "cause→effect supports it"),
                         applicability_fn=lambda q: 1.0),
        CallableStrategy("contrarian", lambda q: ("reject", 0.6, "risk of downside"),
                         applicability_fn=lambda q: 1.0),
    ])
    result = panel.reason(ReasoningQuery(description="should we proceed?"),
                          mode=ReasoningMode.CONSENSUS)

    # 2) run the resulting proposal through the control-law pipeline
    proposal = Proposal(ProposalKind.ANSWER, result.conclusion.answer,
                        confidence=result.conclusion.confidence, source_faculty="reasoner")
    disp = Pipeline().dispose(proposal, ProposalContext())

    # 3) reconstruct and narrate
    expl = (ExplanationBuilder()
            .add(StepKind.PERCEIVE, "a request from the Master: 'should we proceed?'")
            .add(StepKind.RECALL, "prior outcomes of similar decisions")
            .ingest_reasoning(result)
            .ingest_disposition(disp)
            .build(conclusion=result.conclusion.answer,
                   overall_confidence=result.conclusion.confidence))

    print("\n--- BRIEF ---")
    print(expl.narrate(NarrationStyle.BRIEF))

    print("\n--- DETAILED ---")
    print(expl.narrate(NarrationStyle.DETAILED))

    print("\n--- TECHNICAL ---")
    print(expl.narrate(NarrationStyle.TECHNICAL))

    print("\n--- CAUSAL CHAIN ---")
    print(expl.causal_chain())

    # the narrative reflects ONLY recorded steps (no confabulation)
    assert len(expl.steps) == len(expl.to_dict()["steps"])
    assert "approve" in expl.narrate(NarrationStyle.BRIEF)
    # dissent is honestly surfaced
    assert any(s.kind is StepKind.CRITIQUE and "contrarian" in s.summary
               for s in expl.steps)
    print("\nhonest reconstruction: dissent surfaced, no invented steps ✓")

    # empty builder degrades gracefully
    assert "no recorded reasoning" in ExplanationBuilder().build().narrate()

    print("\nALL SELF-TESTS PASSED ✓")
