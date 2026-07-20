"""NYXARA · mind/strategic.py — the Strategic Intelligence faculty (Level 4).

A high-signal analytical engine. Given a problem, NYXARA reasons like a strategist
rather than a chatbot — truth over comfort, first principles over surface patterns —
and returns one structured analysis in a fixed six-part framework:

    1. Direct Answer        — the core judgment, first, in one sentence
    2. Reality Check        — validate (or refute) the premise the problem rests on
    3. Key Weaknesses       — logical flaws, scalability limits, real-world constraints,
                              failure scenarios, unintended consequences
    4. Root Cause           — the underlying driver beneath the surface symptom
    5. Optimized Solution   — the leverage point, not a comfortable compromise
    6. Execution Steps      — concrete, ordered, practical moves

It *composes* existing faculties rather than duplicating them: the
:class:`~nyxara.growth.scientist.Scientist` stress-tests any testable premise (so the
Reality Check is grounded, not asserted), and the
:class:`~nyxara.mind.role_council.RoleCouncil` supplies the Critic / Security Officer /
Engineer / Strategist lenses that surface weaknesses. The
:class:`~nyxara.memory.self_model.SelfModel` keeps it honest: where she is weak or could
hallucinate, confidence drops and the gap is surfaced, never hidden (Rule 6).

This is a pure *analysis* faculty — it proposes structured reasoning and takes no world
actions, so it never reaches around the control law. Everything degrades gracefully:
with no LLM, no council and no scientist it still returns a complete analysis, so it runs
fully offline (and in CI).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = [
    "StrategicIntelligence",
    "StrategicAnalysis",
    "Weakness",
    "PremiseCheck",
    "Soundness",
    "WeaknessKind",
]


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class Soundness(str, Enum):
    """How well the problem's central premise survives scrutiny."""
    SOUND = "sound"
    QUESTIONABLE = "questionable"
    FALSE = "false"


class WeaknessKind(str, Enum):
    """The category of a discovered weakness (the stress-test axes)."""
    LOGIC = "logic"
    SCALABILITY = "scalability"
    REAL_WORLD_CONSTRAINT = "real_world_constraint"
    FAILURE_SCENARIO = "failure_scenario"
    UNINTENDED_CONSEQUENCE = "unintended_consequence"


# Which council role illuminates which weakness axis.
_ROLE_TO_KIND: Dict[str, WeaknessKind] = {
    "Critic": WeaknessKind.LOGIC,
    "Security Officer": WeaknessKind.FAILURE_SCENARIO,
    "Strategist": WeaknessKind.UNINTENDED_CONSEQUENCE,
    "Engineer": WeaknessKind.REAL_WORLD_CONSTRAINT,
}


# --------------------------------------------------------------------------- #
# Domain objects
# --------------------------------------------------------------------------- #
@dataclass
class PremiseCheck:
    """The verdict on the assumption a problem rests on."""
    premise: str
    verdict: Soundness = Soundness.QUESTIONABLE
    correction: str = ""
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "premise": self.premise,
            "verdict": self.verdict.value,
            "correction": self.correction,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class Weakness:
    """One discovered weakness, categorised and severity-scored."""
    issue: str
    kind: WeaknessKind = WeaknessKind.LOGIC
    severity: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue": self.issue,
            "kind": self.kind.value,
            "severity": round(float(self.severity), 3),
        }


@dataclass
class StrategicAnalysis:
    """One complete six-part strategic analysis of a problem."""
    problem: str
    direct_answer: str = ""
    reality_check: Optional[PremiseCheck] = None
    key_weaknesses: List[Weakness] = field(default_factory=list)
    root_cause: str = ""
    optimized_solution: str = ""
    execution_steps: List[str] = field(default_factory=list)
    hidden_assumptions: List[str] = field(default_factory=list)
    worst_case: str = ""
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "direct_answer": self.direct_answer,
            "reality_check": self.reality_check.to_dict() if self.reality_check else None,
            "key_weaknesses": [w.to_dict() for w in self.key_weaknesses],
            "root_cause": self.root_cause,
            "optimized_solution": self.optimized_solution,
            "execution_steps": list(self.execution_steps),
            "hidden_assumptions": list(self.hidden_assumptions),
            "worst_case": self.worst_case,
            "confidence": round(float(self.confidence), 3),
            "timestamp": self.timestamp,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# --------------------------------------------------------------------------- #
# Strategic Intelligence
# --------------------------------------------------------------------------- #
_ASSUMPTION_MARKERS = (
    "since", "because", "given that", "assuming", "obviously", "everyone knows",
    "always", "never", "must", "clearly", "of course",
)
_EQUATION = re.compile(r"\d+(?:\s*[-+*/]\s*\d+)+\s*=\s*\d+")
_CLAUSE = re.compile(
    r"(?:since|because|given that|assuming(?: that)?)\s+(.+?)(?:[,;.]|$)",
    re.IGNORECASE,
)


class StrategicIntelligence:
    """Produce a structured strategic analysis of a problem.

    Parameters
    ----------
    reasoner:    the mind's reasoner; its ``.llm`` is borrowed for synthesis (optional).
    council:     a :class:`RoleCouncil` whose role lenses surface weaknesses (optional).
    scientist:   a :class:`Scientist` used to stress-test the premise (optional).
    world_model: consulted for the worst-case / failure analysis (optional).
    self_model:  keeps confidence honest where she is weak or could hallucinate (optional).
    llm:         an explicit LLM facade; defaults to ``reasoner.llm`` when present.

    Every dependency is optional: with none of them the analysis still completes
    deterministically and offline.
    """

    def __init__(self, *, reasoner: Any = None, council: Any = None,
                 scientist: Any = None, world_model: Any = None,
                 self_model: Any = None, llm: Any = None) -> None:
        self.reasoner = reasoner
        self.council = council
        self.scientist = scientist
        self.world_model = world_model
        self.self_model = self_model
        self.llm = llm if llm is not None else getattr(reasoner, "llm", None)
        self._analyses: List[StrategicAnalysis] = []

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #
    def analyze(self, problem: str) -> StrategicAnalysis:
        """Run the full six-part strategic analysis. Always returns an analysis."""
        t0 = time.monotonic()
        problem = (problem or "").strip()
        analysis = StrategicAnalysis(problem=problem)
        try:
            analysis.hidden_assumptions = self._first_principles(problem)
            analysis.reality_check = self._reality_check(problem, analysis.hidden_assumptions)
            analysis.key_weaknesses = self._weakness_pass(problem)
            self._synthesise(problem, analysis)
            analysis.worst_case = self._worst_case(problem, analysis.key_weaknesses)
            analysis.confidence = self._confidence(problem, analysis)
        except Exception as exc:  # noqa: BLE001 — analysis is advisory, never fatal
            analysis.direct_answer = analysis.direct_answer or (
                f"Unable to complete the analysis cleanly ({exc}); treat with caution.")
            analysis.confidence = min(analysis.confidence, 0.3)
        analysis.elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._analyses.append(analysis)
        return analysis

    # accept the spec's verb as an alias for the method name
    def strategize(self, problem: str) -> StrategicAnalysis:
        return self.analyze(problem)

    def all_analyses(self) -> List[StrategicAnalysis]:
        return list(self._analyses)

    # ---------------------------------------------------------------------- #
    # 0. First principles — surface the hidden assumptions
    # ---------------------------------------------------------------------- #
    def _first_principles(self, problem: str) -> List[str]:
        """Strip the problem to the premises it silently assumes."""
        found: List[str] = []
        low = problem.lower()
        for clause in _CLAUSE.findall(problem):
            clause = clause.strip()
            if clause and clause not in found:
                found.append(clause)
        for marker in _ASSUMPTION_MARKERS:
            if marker in low and not any(marker in f.lower() for f in found):
                found.append(f"the problem assumes '{marker}' holds without proof")
        if not found and problem:
            found.append("the problem is framed as well-posed; that framing is itself untested")
        return found[:5]

    # ---------------------------------------------------------------------- #
    # 2. Reality Check — stress-test the premise (reuse the Scientist)
    # ---------------------------------------------------------------------- #
    def _reality_check(self, problem: str, assumptions: List[str]) -> PremiseCheck:
        """Validate the central premise. Reuses the Scientist for testable claims."""
        claim = self._extract_testable_claim(problem, assumptions)
        if claim and self.scientist is not None:
            verdict, correction, conf = self._investigate(claim)
            return PremiseCheck(premise=claim, verdict=verdict,
                                correction=correction, confidence=conf)
        # heuristic fallback: assumption markers imply an untested premise
        premise = assumptions[0] if assumptions else (problem or "the problem as stated")
        has_marker = any(m in problem.lower() for m in _ASSUMPTION_MARKERS)
        verdict = Soundness.QUESTIONABLE if has_marker else Soundness.SOUND
        correction = (
            "An unproven premise is load-bearing here — verify it before committing."
            if has_marker else
            "No obviously false premise detected, but treat unstated assumptions as risks.")
        return PremiseCheck(premise=premise, verdict=verdict, correction=correction,
                            confidence=0.5)

    def _extract_testable_claim(self, problem: str, assumptions: List[str]) -> str:
        """Pull a concretely checkable claim out of the problem, if one is embedded."""
        eq = _EQUATION.search(problem)
        if eq:
            return f"Is {eq.group(0).strip()}?"
        for a in assumptions:
            if _EQUATION.search(a):
                return f"Is {_EQUATION.search(a).group(0).strip()}?"
        return ""

    def _investigate(self, claim: str) -> tuple[Soundness, str, float]:
        """Run the Scientist on a claim and map its verdict to soundness."""
        try:
            report = self.scientist.investigate(claim)
            conclusion = getattr(report, "conclusion", None)
            if conclusion is None:
                return Soundness.QUESTIONABLE, "Inconclusive on investigation.", 0.4
            verdict = getattr(conclusion.verdict, "value", str(conclusion.verdict))
            reasoning = getattr(conclusion, "reasoning", "") or ""
            conf = float(getattr(conclusion, "confidence", 0.5))
            if verdict == "refuted":
                return Soundness.FALSE, reasoning or "The premise is false.", conf
            if verdict == "supported":
                return Soundness.SOUND, reasoning or "The premise holds.", conf
            return Soundness.QUESTIONABLE, reasoning or "The premise is unverified.", conf
        except Exception:  # noqa: BLE001 — the scientist is a capability, never required
            return Soundness.QUESTIONABLE, "Could not test the premise.", 0.4

    # ---------------------------------------------------------------------- #
    # 3. Key Weaknesses — multi-lens stress test (reuse the RoleCouncil)
    # ---------------------------------------------------------------------- #
    def _weakness_pass(self, problem: str) -> List[Weakness]:
        """Surface weaknesses through the council's adversarial lenses."""
        weaknesses = self._council_weaknesses(problem)
        if not weaknesses:
            weaknesses = self._heuristic_weaknesses(problem)
        weaknesses.sort(key=lambda w: w.severity, reverse=True)
        return weaknesses[:5]

    def _council_weaknesses(self, problem: str) -> List[Weakness]:
        if self.council is None:
            return []
        gather = getattr(self.council, "_gather_votes", None)
        if not callable(gather):
            return []
        try:
            votes = gather(problem) or []
        except Exception:  # noqa: BLE001
            return []
        out: List[Weakness] = []
        for vote in votes:
            member = getattr(vote, "member", "")
            kind = _ROLE_TO_KIND.get(member)
            if kind is None:  # only the adversarial / constraint lenses
                continue
            answer = (getattr(vote, "answer", "") or "").strip()
            if not answer:
                continue
            severity = round(min(0.95, 0.4 + float(getattr(vote, "weight", 0.5)) * 0.5), 3)
            out.append(Weakness(issue=answer, kind=kind, severity=severity))
        return out

    def _heuristic_weaknesses(self, problem: str) -> List[Weakness]:
        """A deterministic, offline weakness set — always categorised, never empty."""
        subject = (problem[:60] + "…") if len(problem) > 60 else (problem or "the plan")
        return [
            Weakness(
                issue=f"Logic: '{subject}' may rest on unstated or untested assumptions.",
                kind=WeaknessKind.LOGIC, severity=0.6),
            Weakness(
                issue=f"Scalability: what works small may not hold as '{subject}' grows.",
                kind=WeaknessKind.SCALABILITY, severity=0.55),
            Weakness(
                issue="Real-world constraint: time, cost, or capability limits are unmodelled.",
                kind=WeaknessKind.REAL_WORLD_CONSTRAINT, severity=0.5),
            Weakness(
                issue="Failure scenario: no defined behaviour when the core assumption breaks.",
                kind=WeaknessKind.FAILURE_SCENARIO, severity=0.5),
        ]

    # ---------------------------------------------------------------------- #
    # 1/4/5. Direct answer, root cause, optimised solution, execution steps
    # ---------------------------------------------------------------------- #
    def _synthesise(self, problem: str, analysis: StrategicAnalysis) -> None:
        """Fill direct_answer, root_cause, optimized_solution, execution_steps."""
        if self._llm_available() and self._synthesise_llm(problem, analysis):
            return
        self._synthesise_offline(problem, analysis)

    def _synthesise_llm(self, problem: str, analysis: StrategicAnalysis) -> bool:
        """Best-effort LLM synthesis into the framework's fields. Returns success."""
        weak = "; ".join(w.issue for w in analysis.key_weaknesses) or "none surfaced"
        premise = analysis.reality_check.premise if analysis.reality_check else problem
        prompt = (
            f"Problem:\n{problem}\n\n"
            f"Central premise under test: {premise}\n"
            f"Known weaknesses: {weak}\n\n"
            "Return strict JSON with keys: direct_answer (one decisive sentence), "
            "root_cause (the underlying driver), optimized_solution (the leverage point), "
            "execution_steps (an array of 3-6 concrete ordered steps)."
        )
        try:
            data = self.llm.generate_json(prompt, system=_SYNTHESIS_SYSTEM,
                                          temperature=0.3, max_tokens=600)
            if not isinstance(data, dict):
                return False
            analysis.direct_answer = str(data.get("direct_answer", "")).strip()
            analysis.root_cause = str(data.get("root_cause", "")).strip()
            analysis.optimized_solution = str(data.get("optimized_solution", "")).strip()
            steps = data.get("execution_steps", [])
            if isinstance(steps, list):
                analysis.execution_steps = [str(s).strip() for s in steps if str(s).strip()]
            return bool(analysis.direct_answer and analysis.optimized_solution
                        and analysis.execution_steps)
        except Exception:  # noqa: BLE001
            return False

    def _synthesise_offline(self, problem: str, analysis: StrategicAnalysis) -> None:
        """Deterministic synthesis — structurally faithful, no LLM required."""
        rc = analysis.reality_check
        top = analysis.key_weaknesses[0] if analysis.key_weaknesses else None

        if rc is not None and rc.verdict is Soundness.FALSE:
            analysis.direct_answer = (
                f"No — the premise is false: {rc.correction}".rstrip("."))
        elif rc is not None and rc.verdict is Soundness.QUESTIONABLE:
            analysis.direct_answer = (
                "Conditionally — the load-bearing premise is unproven and must be "
                "verified before committing.")
        else:
            analysis.direct_answer = (
                "Proceed, but only after the weaknesses below are mitigated.")

        analysis.root_cause = (
            f"The real driver is {top.issue}" if top else
            "The surface problem masks an unvalidated assumption about how the system behaves."
        ).rstrip(".") + "."

        analysis.optimized_solution = (
            "Attack the root cause directly: prove or kill the central premise first, "
            "then design for the highest-severity failure mode rather than the happy path."
        )

        steps: List[str] = []
        if rc is not None:
            steps.append(f"Validate the premise: {rc.premise}")
        if top is not None:
            steps.append(f"Mitigate the top weakness — {top.issue}")
        steps.append("Build the smallest test that could falsify the plan; measure, don't assume.")
        steps.append("Define the failure behaviour explicitly before scaling.")
        steps.append("Decide on the evidence, and reverse cheaply if it disconfirms.")
        analysis.execution_steps = steps

    # ---------------------------------------------------------------------- #
    # Worst case + calibrated confidence
    # ---------------------------------------------------------------------- #
    def _worst_case(self, problem: str, weaknesses: List[Weakness]) -> str:
        failures = [w for w in weaknesses
                    if w.kind in (WeaknessKind.FAILURE_SCENARIO,
                                  WeaknessKind.UNINTENDED_CONSEQUENCE)]
        driver = failures[0].issue if failures else (
            weaknesses[0].issue if weaknesses else "the core assumption fails silently")
        return (f"Worst case: {driver} compounds undetected until it is expensive or "
                "irreversible to correct.")

    def _confidence(self, problem: str, analysis: StrategicAnalysis) -> float:
        conf = 0.5
        rc = analysis.reality_check
        if rc is not None:
            conf += {Soundness.SOUND: 0.15, Soundness.QUESTIONABLE: -0.05,
                     Soundness.FALSE: 0.1}[rc.verdict]  # certainty it's false is still signal
        # heavier / more numerous weaknesses lower confidence in any clean answer
        conf -= 0.05 * sum(w.severity for w in analysis.key_weaknesses)
        # honesty: drop where she is weak or could hallucinate (Rule 6)
        if self.self_model is not None:
            try:
                risk, _domains = self.self_model.hallucination_risk(problem)
                conf -= 0.3 * float(risk)
            except Exception:  # noqa: BLE001
                pass
        # never claim certainty
        return round(max(0.05, min(0.95, conf)), 3)

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #
    def _llm_available(self) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "native"
        except Exception:  # noqa: BLE001
            return False


_SYNTHESIS_SYSTEM = (
    "You are NYXARA's strategic intelligence faculty. Optimise for truth, depth and "
    "usefulness, not comfort. Answer directly first. Use first-principles reasoning. "
    "Be concise and high-signal. Return only the requested JSON."
)


# --------------------------------------------------------------------------- #
# Hermetic self-test — no LLM, no network, no external deps.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    si = StrategicIntelligence()  # fully offline, no dependencies

    a = si.analyze("Since 2 + 2 = 5, we should double the marketing budget.")
    assert a.direct_answer, "direct answer must be set"
    assert a.reality_check is not None
    assert a.key_weaknesses, "weaknesses must be non-empty offline"
    assert a.root_cause and a.optimized_solution
    assert a.execution_steps, "execution steps must be non-empty"
    assert 0.0 <= a.confidence <= 0.95, "confidence is calibrated and never certain"
    d = a.to_dict()
    assert set(["direct_answer", "reality_check", "key_weaknesses", "root_cause",
                "optimized_solution", "execution_steps"]).issubset(d)
    assert all(w["kind"] in {k.value for k in WeaknessKind} for w in d["key_weaknesses"])

    # premise stress-test path via a stub scientist
    class _StubConclusion:
        verdict = "refuted"
        confidence = 0.9
        reasoning = "2 + 2 = 4, not 5."

    class _StubReport:
        conclusion = _StubConclusion()

    class _StubScientist:
        def investigate(self, q):
            return _StubReport()

    si2 = StrategicIntelligence(scientist=_StubScientist())
    a2 = si2.analyze("Given that 2 + 2 = 5, raise the target.")
    assert a2.reality_check.verdict is Soundness.FALSE, a2.reality_check.to_dict()
    assert "No" in a2.direct_answer or "false" in a2.direct_answer.lower()

    print("strategic.py self-test passed:", a.to_dict()["direct_answer"])
